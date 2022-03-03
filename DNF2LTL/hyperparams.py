import itertools

## Parameters used to find a procedural representation of a DNF formula
ITERS = 10000
THRESHOLD = 0.5
ALLOWED_PREDS = ['is_previous_observed_max(st, act)',
                  'is_positive_observed(st, act)',
                  'are_leaves_observed(st, act)',
                  'are_roots_observed(st, act)',
                  'is_previous_observed_positive(st, act)',
                  'is_previous_observed_min(st, act)',
                  'is_previous_observed_max_nonleaf(st, act)',
                  'is_previous_observed_max_leaf(st, act)',
                  'is_previous_observed_max_root(st, act)',
                  'is_previous_observed_max_level(st, act,  1 )',
                  'is_previous_observed_max_level(st, act,  2 )',
                  'is_previous_observed_max_level(st, act,  3 )',
                  'is_previous_observed_min_level(st, act,  1 )',
                  'is_previous_observed_min_level(st, act,  2 )',
                  'is_previous_observed_min_level(st, act,  3 )',
                  'termination_return(st, act,  -30 )',
                  'termination_return(st, act,  -25 )',
                  'termination_return(st, act,  -15 )',
                  'termination_return(st, act,  -10 )',
                  'termination_return(st, act,  0 )',
                  'termination_return(st, act,  10 )',
                  'termination_return(st, act,  15 )',
                  'termination_return(st, act,  25 )',
                  'termination_return(st, act,  30 )',
                  'is_max_in_branch(st, act)',
                  'are_branch_leaves_observed(st, act)']
combs = list(itertools.combinations(ALLOWED_PREDS, 2))
allowed_combs = ['(' + left + ' or ' + right + ')' for left, right in combs 
                 if (left[:10] != right[:10])]
ALLOWED_PREDS += allowed_combs
REDUNDANT_TYPES = ['all_']

## Parameters simplifying translating a procedural formula into words
CLEANED_PREDS = ['is_previous_observed_max',
                 'is_positive_observed',
                 'are_leaves_observed',
                 'are_roots_observed',
                 'is_previous_observed_positive',
                 'is_previous_observed_min',
                 'is_previous_observed_max_nonleaf',
                 'is_previous_observed_max_leaf',
                 'is_previous_observed_max_root',
                 'is_previous_observed_max_level',
                 'is_previous_observed_max_level',
                 'is_previous_observed_max_level',
                 'is_previous_observed_min_level',
                 'is_previous_observed_min_level',
                 'is_previous_observed_min_level',
                 'observed_count( 1)',
                 'observed_count( 2)',
                 'observed_count( 3)',
                 'observed_count( 4)',
                 'observed_count( 5)',
                 'observed_count( 6)',
                 'observed_count( 7)',
                 'observed_count( 8)',
                 'termination_return( -30)',
                 'termination_return( -25)',
                 'termination_return( -15)',
                 'termination_return( -10)',
                 'termination_return( 0)',
                 'termination_return( 10)',
                 'termination_return( 15)',
                 'termination_return( 25)',
                 'termination_return( 30)',
                 'is_max_in_branch',
                 'are_branch_leaves_observed'
                 'is_previous_observed_sibling',
                 'is_previous_observed_parent']
CLEANED_PREDS_N = ['not(' + c + ')' for c in CLEANED_PREDS]
CLEANED_PREDS += CLEANED_PREDS_N
