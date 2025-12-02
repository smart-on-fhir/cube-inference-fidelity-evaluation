# Core imports
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy.stats import poisson
import matplotlib.pyplot as plt

# Optimization and nullspace computation
import pulp
import sympy as sp
from cmdstanpy import CmdStanModel

import itertools
from collections import defaultdict

def parse_cell_indices(cell_indices_str: str) -> List[int]:
    """Helper function to safely parse cell indices string."""
    try:
        if pd.isna(cell_indices_str) or cell_indices_str.strip() == "":
            return []
        return [int(x) for x in cell_indices_str.strip().split(';') if x.strip() != '']
    except:
        return []

def line_level_to_atomic_cells(df_line_level: pd.DataFrame, 
                              demographic_cols: List[str] = None) -> pd.DataFrame:
    if demographic_cols is None:
        demographic_cols = list(df_line_level.columns)
    
    # Get all unique combinations (atomic cells)
    unique_combinations = df_line_level[demographic_cols].drop_duplicates().reset_index(drop=True)
    
    # Create cell labels
    labels = []
    for _, row in unique_combinations.iterrows():
        label_parts = [f"{col}={row[col]}" for col in demographic_cols]
        labels.append("_".join(label_parts))
    
    # Create cell DataFrame
    cells_df = unique_combinations.copy()
    cells_df.insert(0, 'cell_id', range(len(cells_df)))
    cells_df.insert(1, 'label', labels)
    
    return cells_df

def line_level_to_powerset(df_line_level: pd.DataFrame, 
                          demographic_cols: List[str] = None,
                          min_count_threshold: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if demographic_cols is None:
        demographic_cols = list(df_line_level.columns)
    
    print(f"Converting line-level data to powerset using columns: {demographic_cols}")
    print(f"Line-level data shape: {df_line_level.shape}")
    
    # Generate atomic cells
    cells_df = line_level_to_atomic_cells(df_line_level, demographic_cols)
    J = len(cells_df)
    
    # Create mapping from combination to cell_id
    cell_lookup = {}
    for _, cell_row in cells_df.iterrows():
        key = tuple(cell_row[col] for col in demographic_cols)
        cell_lookup[key] = cell_row['cell_id']
    
    powerset_rows = []
    subset_id = 0
    n_vars = len(demographic_cols)
    
    for r in range(n_vars + 1):
        for combo in itertools.combinations(demographic_cols, r):
            grouping_cols = list(combo)
            
            if len(grouping_cols) == 0:
                # Grand total
                count = len(df_line_level)
                cell_indices = list(range(J))  # All cells
                subset_name = "TOTAL"
                
                # Create row with separate columns for each demographic variable
                row_data = {
                    'subset_id': subset_name,
                    'cell_indices': ';'.join(map(str, cell_indices)),
                    'count': count if count >= min_count_threshold else 'S',
                    'count_actual': count
                }
                # Add demographic columns as NA
                for col in demographic_cols:
                    row_data[col] = pd.NA
                
                powerset_rows.append(row_data)
            else:
                # Group by the selected columns
                grouped = df_line_level.groupby(grouping_cols).size().reset_index(name='count')
                
                for _, group_row in grouped.iterrows():
                    count = group_row['count']
                    
                    # Find which atomic cells belong to this group
                    cell_indices = []
                    group_values = {col: group_row[col] for col in grouping_cols}
                    
                    for _, cell_row in cells_df.iterrows():
                        # Check if this cell matches the group
                        matches = all(cell_row[col] == group_values[col] for col in grouping_cols)
                        if matches:
                            cell_indices.append(cell_row['cell_id'])
                    
                    # Create subset name (kept for backward compatibility)
                    if len(grouping_cols) == 1:
                        subset_name = f"{grouping_cols[0]}_{group_row[grouping_cols[0]]}"
                    else:
                        parts = [f"{col}={group_row[col]}" for col in grouping_cols]
                        subset_name = "_".join(parts)
                    
                    # Apply suppression rule
                    count_val = count if count >= min_count_threshold else 'S'
                    
                    if cell_indices:  # Only add if we have matching cells
                        # Create row with separate columns for each demographic variable
                        row_data = {
                            'subset_id': subset_name,
                            'cell_indices': ';'.join(map(str, cell_indices)),
                            'count': count_val,
                            'count_actual': count
                        }
                        # Add demographic columns: value if in grouping, NA otherwise
                        for col in demographic_cols:
                            if col in grouping_cols:
                                row_data[col] = group_row[col]
                            else:
                                row_data[col] = pd.NA
                        
                        powerset_rows.append(row_data)
    
    powerset_df = pd.DataFrame(powerset_rows)
    
    published_count = sum(1 for _, row in powerset_df.iterrows() 
                         if row['count'] != 'S' and pd.notna(row['count']))
    suppressed_count = len(powerset_df) - published_count
    print(f"  {published_count} published constraints")
    print(f"  {suppressed_count} suppressed constraints")
    
    return cells_df, powerset_df

def build_matrices(df_powerset: pd.DataFrame, J: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    rows_eq, y_eq, rows_ineq, subset_names = [], [], [], []
    
    for _, row in df_powerset.iterrows():
        if 'cells' in row:
            cells = row['cells']
            count = row['count_val'] if 'count_val' in row else row.get('count')
        else:
            cells = parse_cell_indices(row['cell_indices'])
            count = row['count']
            if count == 'S':
                count = None
        
        vec = np.zeros(J, dtype=int)
        if cells:  # Only set if cells is not empty
            vec[cells] = 1
        
        if pd.notna(count) and count is not None:
            try:
                count_float = float(count)
                if not np.isnan(count_float):
                    # Published constraint: exact equality
                    rows_eq.append(vec)
                    y_eq.append(int(count_float))
                    subset_names.append(row['subset_id'])
                    continue
            except (ValueError, TypeError):
                pass
        
        # Suppressed constraint: 0 <= sum <= threshold
        rows_ineq.append(vec)
        subset_names.append(row['subset_id'])
    
    A_eq = np.vstack(rows_eq) if rows_eq else np.zeros((0, J), dtype=int)
    y_eq = np.array(y_eq, dtype=int) if y_eq else np.zeros((0,), dtype=int)
    A_ineq = np.vstack(rows_ineq) if rows_ineq else np.zeros((0, J), dtype=int)
    
    return A_eq, y_eq, A_ineq, subset_names


def ilp_bounds(A_eq: np.ndarray, y_eq: np.ndarray, A_ineq: np.ndarray, 
               J: int, suppress_threshold: int = 9, time_limit: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute exact integer bounds for each cell using ILP.
    
    Parameters:
        A_eq: Equality constraint matrix (published counts)
        y_eq: Published count values
        A_ineq: Inequality constraint matrix (suppressed counts)
        J: Number of atomic cells
        suppress_threshold: Suppression threshold
        time_limit: Solver time limit in seconds
    
    Returns:
        L: lower bounds for each cell
        U: upper bounds for each cell
    """
    L, U = np.zeros(J, dtype=int), np.zeros(J, dtype=int)
    
    # Try available ILP solvers
    solvers_to_try = [
        pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit),
        pulp.GLPK_CMD(msg=False),
        pulp.CPLEX_CMD(msg=False),
        pulp.GUROBI_CMD(msg=False)
    ]
    
    working_solver = None
    for solver in solvers_to_try:
        if solver.available():
            working_solver = solver
            break
    
    # Solve min/max for each cell
    for j in range(J):
        # Minimize cell j
        prob_min = pulp.LpProblem("min_cell", pulp.LpMinimize)
        x = [pulp.LpVariable(f"x_{k}", lowBound=0, cat='Integer') for k in range(J)]
        prob_min += x[j]  # objective
        
        # Equality constraints (always use published constraints)
        for i in range(A_eq.shape[0]):
            prob_min += pulp.lpSum([int(A_eq[i, k]) * x[k] for k in range(J)]) == int(y_eq[i])
        
        # Suppressed inequalities - enforce that suppressed sums <= threshold
        for i in range(A_ineq.shape[0]):
            prob_min += pulp.lpSum([int(A_ineq[i, k]) * x[k] for k in range(J)]) <= suppress_threshold
            prob_min += pulp.lpSum([int(A_ineq[i, k]) * x[k] for k in range(J)]) >= 0
        
        # Solve
        prob_min.solve(working_solver)
        L[j] = int(round(pulp.value(prob_min.objective))) if prob_min.status == 1 else 0
        
        # Maximize cell j
        prob_max = pulp.LpProblem("max_cell", pulp.LpMaximize)
        x2 = [pulp.LpVariable(f"x2_{k}", lowBound=0, cat='Integer') for k in range(J)]
        prob_max += x2[j]  # objective
        
        # Same constraints
        for i in range(A_eq.shape[0]):
            prob_max += pulp.lpSum([int(A_eq[i, k]) * x2[k] for k in range(J)]) == int(y_eq[i])
        
        for i in range(A_ineq.shape[0]):
            prob_max += pulp.lpSum([int(A_ineq[i, k]) * x2[k] for k in range(J)]) <= suppress_threshold
            prob_max += pulp.lpSum([int(A_ineq[i, k]) * x2[k] for k in range(J)]) >= 0
        
        prob_max.solve(working_solver)
        U[j] = int(round(pulp.value(prob_max.objective))) if prob_max.status == 1 else 1000
    
    return L, U

def find_feasible_solution(A_eq: np.ndarray, y_eq: np.ndarray, A_ineq: np.ndarray, 
                          J: int, suppress_threshold: int = 9, 
                          L: Optional[np.ndarray] = None, U: Optional[np.ndarray] = None) -> np.ndarray:
    """Find any feasible integer solution."""
    # Use same solver setup as ilp_bounds
    working_solver = None
    for solver in [pulp.PULP_CBC_CMD(msg=False), pulp.GLPK_CMD(msg=False)]:
        if solver.available():
            working_solver = solver
            break
    
    prob = pulp.LpProblem("feasible", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{k}", lowBound=0, cat='Integer') for k in range(J)]
    
    # Bounds
    if L is not None:
        for k in range(J):
            if np.isfinite(L[k]):
                prob += x[k] >= int(L[k])
    if U is not None:
        for k in range(J):
            if np.isfinite(U[k]):
                prob += x[k] <= int(U[k])
    
    # Constraints
    for i in range(A_eq.shape[0]):
        prob += pulp.lpSum([int(A_eq[i, k]) * x[k] for k in range(J)]) == int(y_eq[i])
    for i in range(A_ineq.shape[0]):
        prob += pulp.lpSum([int(A_ineq[i, k]) * x[k] for k in range(J)]) <= suppress_threshold
        prob += pulp.lpSum([int(A_ineq[i, k]) * x[k] for k in range(J)]) >= 0
    
    # Minimize sum as arbitrary objective
    prob += pulp.lpSum(x)
    
    prob.solve(working_solver)
    if prob.status != 1:
        raise RuntimeError(f"No feasible solution found (status={prob.status})")
    
    return np.array([int(round(pulp.value(xk))) for xk in x], dtype=int)

def integer_nullspace_basis(A_eq: np.ndarray) -> np.ndarray:
    J = A_eq.shape[1]
    if A_eq.shape[0] == 0:
        return np.eye(J, dtype=int)
    
    M = sp.Matrix(A_eq.tolist())
    null = M.nullspace()
    
    if len(null) == 0:
        return np.zeros((J, 0), dtype=int)
    
    cols = []
    for v in null:
        denoms = [sp.Rational(val).q for val in v]
        lcm = 1
        for d in denoms:
            lcm = int(np.lcm(lcm, int(d)))
        
        vec_int = np.array([int(sp.Rational(val) * lcm) for val in v], dtype=int)
        
        nonzero = np.abs(vec_int[np.abs(vec_int) > 0])
        if len(nonzero) > 0:
            g = int(np.gcd.reduce(nonzero))
            if g > 1:
                vec_int = vec_int // g
        
        cols.append(vec_int)
    
    return np.column_stack(cols).astype(int)


# Stan model
STAN_MODEL = r"""
data {
  int<lower=1> J;            // number of atomic cells
  int<lower=1> K;            // number of predictors 
  matrix[J, K] X;            // design matrix
  int<lower=0> nP;           // number of published subsets
  int<lower=1> maxP;         // max size of published constraint
  array[nP] int<lower=1> P_sizes;
  array[nP, maxP] int P_idx; // padded indices (1-based)
  array[nP] int yP;
  int<lower=0> nQ;          // number of suppressed subsets
  int<lower=1> maxQ;         // max size of suppressed constraint
  array[nQ] int<lower=1> Q_sizes;
  array[nQ, maxQ] int Q_idx;
  int<lower=0> suppress_threshold;
}
parameters {
  vector[K] beta;
}
transformed parameters {
  vector[J] log_lambda = X * beta;
  vector[J] lambda = exp(log_lambda);
}
model {
  // Priors
  beta ~ normal(0, 2);
  
  // Published exact subsets
  for (i in 1:nP) {
    real s = 0;
    for (k in 1:P_sizes[i]) {
      s += lambda[P_idx[i, k]];
    }
    yP[i] ~ poisson(s);
  }
  
  // Suppressed subsets: censored likelihood
  for (i in 1:nQ) {
    real s = 0;
    for (k in 1:Q_sizes[i]) {
      s += lambda[Q_idx[i, k]];
    }
    target += poisson_lcdf(suppress_threshold | s);
  }
}
generated quantities {
  vector[J] lambda_out = lambda;
}
"""

def write_stan_model(outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "reconstruction_model.stan")
    with open(path, "w") as f:
        f.write(STAN_MODEL)
    return path

def build_design_matrix(J: int) -> Tuple[np.ndarray, List[str]]:
    return np.eye(J), [f"cell_{i}" for i in range(J)]

    
def fit_stan_model(outdir: str, X: np.ndarray, A_eq: np.ndarray, y_eq: np.ndarray, 
                   A_ineq: np.ndarray, suppress_threshold: int, df_powerset: pd.DataFrame,
                   iter_sampling: int = 800, iter_warmup: int = 400, chains: int = 4) -> np.ndarray:
    """
    Fit Stan model and return posterior lambda draws.
    """
    J, K = X.shape
    
    # Organize data for Stan
    P_rows, Q_rows = [], []
    for _, row in df_powerset.iterrows():
        if 'cells' in row:
            cells = row['cells']
            count_val = row['count_val'] if 'count_val' in row else row.get('count')
        else:
            cells = parse_cell_indices(row['cell_indices'])
            count_val = row['count']
            if count_val == 'S':
                count_val = None
        
        if pd.notna(count_val) and count_val is not None and count_val != 'S':
            try:
                count_float = float(count_val)
                if not np.isnan(count_float):
                    P_rows.append((row['subset_id'], cells, int(count_float)))
                else:
                    Q_rows.append((row['subset_id'], cells))
            except (ValueError, TypeError):
                Q_rows.append((row['subset_id'], cells))
        else:
            Q_rows.append((row['subset_id'], cells))
    
    nP, nQ = len(P_rows), len(Q_rows)
    
    # Handle edge cases
    if nP == 0:
        P_sizes = []
        maxP = 1
        P_idx = np.empty((0, 1), dtype=int)
        yP = np.array([], dtype=int)
    else:
        P_sizes = [len(cells) for _, cells, _ in P_rows]
        maxP = max(P_sizes)
        P_idx = np.ones((nP, maxP), dtype=int)
        yP = np.zeros(nP, dtype=int)
        for i, (_, cells, cnt) in enumerate(P_rows):
            for k, c in enumerate(cells):
                P_idx[i, k] = c + 1  # Stan uses 1-based indexing
            yP[i] = cnt
    
    if nQ == 0:
        Q_sizes = []
        maxQ = 1
        Q_idx = np.empty((0, 1), dtype=int)
    else:
        Q_sizes = [len(cells) for _, cells in Q_rows]
        maxQ = max(Q_sizes)
        Q_idx = np.ones((nQ, maxQ), dtype=int)
        for i, (_, cells) in enumerate(Q_rows):
            for k, c in enumerate(cells):
                Q_idx[i, k] = c + 1
    
    # Prepare Stan data
    stan_data = {
        'J': J, 'K': K, 'X': X.tolist(),
        'nP': nP, 'maxP': maxP, 'P_sizes': P_sizes, 'P_idx': P_idx.tolist(), 'yP': yP.tolist(),
        'nQ': nQ, 'maxQ': maxQ, 'Q_sizes': Q_sizes, 'Q_idx': Q_idx.tolist(),
        'suppress_threshold': suppress_threshold
    }
    
    # Compile and fit
    stan_file = write_stan_model(outdir)
    print("Compiling Stan model...")
    model = CmdStanModel(stan_file=stan_file)
    
    print("Fitting model...")
    fit = model.sample(data=stan_data, chains=chains, 
                      iter_sampling=iter_sampling, iter_warmup=iter_warmup)
    
    # Extract lambda draws
    draws = fit.draws_pd()
    lambda_cols = [f'lambda_out[{i}]' for i in range(1, J+1)]
    lambda_draws = draws[lambda_cols].values
    
    # Save results
    np.savez_compressed(os.path.join(outdir, "lambda_draws.npz"), lambda_draws=lambda_draws)
    print(f"Saved {lambda_draws.shape[0]} lambda draws to {outdir}")
    
    return lambda_draws

def integer_constrained_sampling(lambda_draws: np.ndarray, A_eq: np.ndarray, y_eq: np.ndarray,
                                A_ineq: np.ndarray, L: np.ndarray, U: np.ndarray,
                                suppress_threshold: int, mh_iters: int = 5000, thin: int = 10) -> np.ndarray:
    """
    Sample integer counts that satisfy constraints exactly.
    """
    n_draws, J = lambda_draws.shape
    print(f"Integer sampling for {n_draws} posterior draws...")
    
    # Compute nullspace basis
    B = integer_nullspace_basis(A_eq)
    r = B.shape[1]
    
    if r == 0:
        n0 = find_feasible_solution(A_eq, y_eq, A_ineq, J, suppress_threshold, L, U)
        return np.tile(n0, (n_draws, 1))
    
    all_samples = []
    
    for t_idx in range(n_draws):
        lam = lambda_draws[t_idx, :]
        
        # Find feasible starting point
        n0 = find_feasible_solution(A_eq, y_eq, A_ineq, J, suppress_threshold, L, U)
        
        # MH sampling on nullspace coordinates
        t_vec = np.zeros(r, dtype=int)
        n_curr = n0.copy()
        
        def log_likelihood(n):
            return np.sum(poisson.logpmf(n, lam))
        
        logp_curr = log_likelihood(n_curr)
        samples = []
        accepted = 0
        
        for it in range(mh_iters):
            k = np.random.randint(r)
            step = np.random.choice([-1, 1])
            t_prop = t_vec.copy()
            t_prop[k] += step
            
            n_prop = n0 + B @ t_prop
            
            # Check bounds and constraints
            if (np.all(n_prop >= L) and np.all(n_prop <= U) and 
                np.all(A_ineq @ n_prop <= suppress_threshold) and
                np.all(A_ineq @ n_prop >= 0)):
                
                logp_prop = log_likelihood(n_prop)
                log_alpha = logp_prop - logp_curr
                
                if np.log(np.random.rand()) < log_alpha:
                    t_vec = t_prop
                    n_curr = n_prop
                    logp_curr = logp_prop
                    accepted += 1
            
            if it % thin == 0:
                samples.append(n_curr.copy())
        
        acc_rate = accepted / mh_iters
        print(f"Draw {t_idx+1}/{n_draws}: acceptance rate {acc_rate:.3f}")
        all_samples.extend(samples)
    
    return np.array(all_samples)


####### Visualization #######
def create_powerset_summary_table(results: Dict):
    
    powerset_df = results['powerset_df'].copy()
    reconstruction_results = results.get('results', {})
    
    standard_cols = ['subset_id', 'cell_indices', 'count', 'count_actual']
    demographic_cols = [col for col in powerset_df.columns if col not in standard_cols]
    
    summary_rows = []
    
    for idx, row in powerset_df.iterrows():
        subset_id = row['subset_id']
        count = row['count']
        count_actual = row.get('count_actual', None)
        
        # Determine status
        if count != 'S' and pd.notna(count):
            status = 'Published'
            value_display = f"{int(count)}"
            bounds_display = ""
            posterior_stats = ""
            posterior_std = ""
        else:
            reconstruction = reconstruction_results.get(idx, None)
            if reconstruction is not None:
                bounds = reconstruction.get('bounds', [None, None])
                std_val = reconstruction.get('std', 0)
                if bounds[0] == bounds[1]:
                    status = 'Deterministic'
                    value_display = f"{bounds[0]}"
                    bounds_display = f"[{bounds[0]}, {bounds[1]}]"
                    posterior_stats = ""
                    posterior_std = "0.0"
                else:
                    status = 'Uncertain'
                    mean_val = reconstruction.get('mean', 0)
                    median_val = reconstruction.get('median', 0)
                    ci_95 = reconstruction.get('ci_95', [0, 0])
                    value_display = f"μ={mean_val:.1f}"
                    bounds_display = f"[{bounds[0]}, {bounds[1]}]"
                    posterior_stats = f"95%CI: [{ci_95[0]:.1f}, {ci_95[1]:.1f}]"
                    posterior_std = f"{std_val:.2f}"
            else:
                status = 'Suppressed'
                value_display = 'S'
                bounds_display = ""
                posterior_stats = ""
                posterior_std = ""
        
        row_data = {
            'Subset': subset_id,
            'Status': status,
            'Value': value_display,
            'Bounds': bounds_display,
            'Posterior_Std': posterior_std,
            'Posterior_Stats': posterior_stats,
            'Actual': count_actual if count_actual is not None else ""
        }
        
        # Add demographic columns
        for col in demographic_cols:
            row_data[col] = row[col]
        
        summary_rows.append(row_data)
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Sort by status
    status_order = {'Published': 0, 'Deterministic': 1, 'Uncertain': 2, 'Suppressed': 3}
    summary_df['status_order'] = summary_df['Status'].map(status_order)
    summary_df = summary_df.sort_values(['status_order', 'Subset']).drop('status_order', axis=1)
    
    return summary_df

def display_complete_powerset_table(results: Dict):
    summary_df = create_powerset_summary_table(results)
    # Organize by constraint complexity/type
    categories = {
        'Grand Total': [],
        'Single Variable Marginals': [],
        'Two-Variable Intersections': [],
        'Three-Variable Atomic Cells': []
    }
    
    for idx, row in summary_df.iterrows():
        subset_id = row['Subset']
        
        if subset_id == 'TOTAL':
            categories['Grand Total'].append(row)
        elif '=' not in subset_id:
            # Single variable marginals like "Sex_M", "Diagnosis_COVID"
            categories['Single Variable Marginals'].append(row)
        elif subset_id.count('=') == 2:
            # Two-variable intersections like "Sex=M_Diagnosis=COVID"
            categories['Two-Variable Intersections'].append(row)
        elif subset_id.count('=') == 3:
            # Three-variable atomic cells like "Sex=M_Diagnosis=COVID_Race=Black"
            categories['Three-Variable Atomic Cells'].append(row)
    
    # Display each category
    for category, rows in categories.items():
        if rows:
            print(f"\n{category} ({len(rows)} constraints):")
            print("-" * 40)
            
            for row in rows:
                status_symbol = {
                    'Published': '✓',
                    'Deterministic': '◉',
                    'Uncertain': '◯',
                    'Suppressed': '✗'
                }
                
                symbol = status_symbol.get(row['Status'], '?')
                
                if row['Status'] == 'Published':
                    print(f"  {symbol} {row['Subset']:<25} = {row['Value']:<10} (actual: {row['Actual']})")
                elif row['Status'] == 'Deterministic':
                    print(f"  {symbol} {row['Subset']:<25} = {row['Value']:<10} (exact, actual: {row['Actual']})")
                else:  # Uncertain
                    print(f"  {symbol} {row['Subset']:<25} = {row['Value']:<10} {row['Bounds']:<12} (actual: {row['Actual']})")
    
    print(f"\nLEGEND:")
    print("✓ Published (count > suppression threshold)")
    print("◉ Suppressed but deterministic") 
    print("◯ Suppressed with uncertainty")
    
    return summary_df

def run_complete_bayesian_reconstruction(
    df_line_level: pd.DataFrame = None,
    line_level_csv: str = None, 
    demographic_cols: List[str] = None,
    suppress_threshold: int = 9,
    iter_sampling: int = 200, 
    iter_warmup: int = 100,
    chains: int = 2,
    mh_iters: int = 2000,
) -> Dict:
    
    # Load line-level data
    if df_line_level is None and line_level_csv is None:
        raise ValueError("Either df_line_level or line_level_csv must be provided")
    
    if df_line_level is None:
        print(f"Loading line-level data from {line_level_csv}...")
        df_line_level = pd.read_csv(line_level_csv)
    else:
        print("Using provided line-level DataFrame...")
    
    print(f"   {len(df_line_level)} individual records")
    
    if demographic_cols is None:
        demographic_cols = list(df_line_level.columns)
    
    print(f"   Using demographic columns: {demographic_cols}")
    
    # Convert to powerset format
    print("Converting to CUBE format...")
    cells_df, powerset_df = line_level_to_powerset(
        df_line_level, 
        demographic_cols=demographic_cols,
        min_count_threshold=suppress_threshold + 1  # Suppress counts <= threshold
    )
    
    J = len(cells_df)

    # Build constraint system  
    print("📋 Building constraint matrices...")
    A_eq, y_eq, A_ineq, subset_names = build_matrices(powerset_df, J)
    print(f"   {A_eq.shape[0]} equality constraints (published)")
    print(f"   {A_ineq.shape[0]} inequality constraints (suppressed)")
    
    # Compute bounds for atomic cells
    print("🔢 Computing atomic cell bounds via ILP...")
    L, U = ilp_bounds(A_eq, y_eq, A_ineq, J, suppress_threshold)

    # Identify ALL suppressed constraints
    print("🔍 Identifying ALL suppressed constraints...")
    suppressed_constraints = []
    suppressed_constraint_info = []
    
    for idx, (_, row) in enumerate(powerset_df.iterrows()):
        if row['count'] == 'S':  # Suppressed constraint
            cell_indices = parse_cell_indices(row['cell_indices'])
            
            # Determine constraint type
            if len(cell_indices) == 1:
                constraint_type = 'atomic'
            else:
                constraint_type = 'aggregate'
            
            constraint_info = {
                'constraint_id': idx,
                'subset_id': row['subset_id'],
                'cell_indices': cell_indices,
                'type': constraint_type,
                'n_cells': len(cell_indices)
            }
            suppressed_constraints.append(idx)
            suppressed_constraint_info.append(constraint_info)
    
    print(f"   Found {len(suppressed_constraints)} suppressed constraints:")
    atomic_count = sum(1 for info in suppressed_constraint_info if info['type'] == 'atomic')
    aggregate_count = sum(1 for info in suppressed_constraint_info if info['type'] == 'aggregate')
    
    # Compute bounds for ALL suppressed constraints
    print("🔢 Computing bounds for ALL suppressed constraints...")
    constraint_bounds = {}
    
    for constraint_info in suppressed_constraint_info:
        constraint_id = constraint_info['constraint_id']
        cell_indices = constraint_info['cell_indices']
        
        if len(cell_indices) == 0:
            # Empty constraint
            constraint_bounds[constraint_id] = [0, 0]
        elif len(cell_indices) == 1:
            # Atomic constraint - use precomputed bounds from ILP
            j = cell_indices[0]
            constraint_bounds[constraint_id] = [int(L[j]), int(U[j])]
        else:
            lower_bound = sum(int(L[j]) for j in cell_indices)
            upper_bound = min(suppress_threshold, sum(int(U[j]) for j in cell_indices))
            constraint_bounds[constraint_id] = [lower_bound, upper_bound]

    # Stan fitting
    print("🎲 Bayesian inference with Stan...")
    outdir = "./reconstruction_results"
    X, col_names = build_design_matrix(J)
    
    lambda_draws = fit_stan_model(outdir, X, A_eq, y_eq, A_ineq, suppress_threshold, 
                                 powerset_df, iter_sampling=iter_sampling, 
                                 iter_warmup=iter_warmup, chains=chains)
    
    # Integer sampling for atomic cells
    print("🔗 Integer-constrained sampling for atomic cells...")
    posterior_samples = integer_constrained_sampling(lambda_draws, A_eq, y_eq, A_ineq, 
                                                   L, U, suppress_threshold, mh_iters=mh_iters)
    
    # Compute posterior samples for ALL suppressed constraints
    print("📊 Computing posterior distributions for ALL suppressed constraints...")
    n_samples = posterior_samples.shape[0]
    results = {}
    
    for constraint_info in suppressed_constraint_info:
        constraint_id = constraint_info['constraint_id']
        subset_id = constraint_info['subset_id']
        cell_indices = constraint_info['cell_indices']
        constraint_type = constraint_info['type']
        bounds = constraint_bounds[constraint_id]
        
        # Compute constraint samples by summing relevant atomic cells
        if len(cell_indices) == 0:
            constraint_samples = np.zeros(n_samples)
        else:
            constraint_samples = np.sum(posterior_samples[:, cell_indices], axis=1)
        
        # Store results
        results[constraint_id] = {
            'constraint_id': constraint_id,
            'subset_id': subset_id,
            'type': constraint_type,
            'cell_indices': cell_indices,
            'n_cells': len(cell_indices),
            'bounds': bounds,
            'mean': float(np.mean(constraint_samples)),
            'median': float(np.median(constraint_samples)),
            'std': float(np.std(constraint_samples)),
            'ci_95': [float(np.percentile(constraint_samples, 2.5)), 
                     float(np.percentile(constraint_samples, 97.5))],
            'samples': constraint_samples
        }
    
    print(f"✅ Successfully reconstructed {len(results)} suppressed constraints!")
    
    return {
        'suppressed_constraints': suppressed_constraints,
        'results': results,
        'atomic_posterior_samples': posterior_samples,  # (n_samples, J) for atomic cells
        'lambda_draws': lambda_draws,
        'atomic_bounds': {'L': L, 'U': U},  # Bounds for atomic cells
        'constraint_bounds': constraint_bounds,  # Bounds for all constraints
        'constraint_info': suppressed_constraint_info,
        'n_atomic_suppressed': atomic_count,
        'n_aggregate_suppressed': aggregate_count,
        'total_suppressed': len(suppressed_constraints),
        'cells_df': cells_df,
        'powerset_df': powerset_df,
        'line_level_df': df_line_level,
        'outdir': outdir
    }