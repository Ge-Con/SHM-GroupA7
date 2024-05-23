def Pr(X):
    """
    This function calculates the prognosability value for a set of HIs.
    Input 'X':  matrix of all extracted HIs (m rows x n columns),
    where one row represents one HI and columns represent timesteps
    Output 'prognosability': scalar, prognosability value for set of HIs,
    ranges from 0 to 1
    """
    M = len(X)
    Nfeatures = X.shape[1]
    top = np.zeros((M, Nfeatures))
    bottom = np.zeros((M, Nfeatures))

    for j in range(M):
        top[j, :] = X[j, -1]
        bottom[j, :] = np.abs(X[j, 0] - X[j, -1])

    prognosability = np.exp(-np.std(top) / np.mean(bottom))

    return prognosability


def Tr(X):
    """
    This function calculates the trendability value for a set of HIs.
    Input 'X':  matrix of all extracted HIs (m rows x n columns),
    where one row represents one HI and columns represent timesteps
    Output 'trendability': scalar, trendability value for set of HIs,
    ranges from 0 to 1
    """
    m, n = X.shape  # m = rows (ie. # of samples), n = columns (ie. # of cycles)
    trendability_feature = np.inf
    trendability_list = []

    # Finding normalised matrix
    scaler = Normalizer()
    X = scaler.fit_transform(X)

    for j in range(m):
        # Obtain (pre-normalised) first vector of measurements
        vector1 = X[j]

        for k in range(m):
            # Obtain (pre-normalised) second vector of measurements
            vector2 = X[k]

            # Check whether two vectors are the same length, ie. if they both experience failure at same cycle
            # If vectors are not same length, reshape/resample them to equal lengths
            if len(vector2) != len(vector1):
                if len(vector2) < len(vector1):
                    vector2 = resample_poly(vector2, len(vector1), len(vector2), window=('kaiser', 5))
                else:
                    vector1 = resample_poly(vector1, len(vector2), len(vector1), window=('kaiser', 5))

            rho = pearsonr(vector1, vector2)[0]
            if math.fabs(rho) < trendability_feature:  # ie. if less than infinity, give new value
                trendability_feature = math.fabs(rho)

            # Add math.fabs(rho) to list
            trendability_list.append(trendability_feature)

    # Return minimum value
    return min(trendability_list)


def Mo_single(X_single) -> float:
    """
    This function calculates the monotonicity value for a single HI.
    Input 'X_single':  row containing an HI (1 rows x n columns),
    where the row represents the HI and columns represent timesteps
    Output 'monotonicity_single': scalar, monotonicity value for a single of HI,
    ranges from 0 to 1
    """
    sum_samples = 0
    for i in range(len(X_single)):
        sum_measurements = 0
        div_sum = 0
        for k in range(len(X_single)):
            sub_sum = 0
            div_sub_sum = 0
            if k > i:
                sub_sum += (k - i) * np.sign(X_single[k] - X_single[i])
                div_sub_sum += k - i
            sum_measurements += sub_sum
            div_sum += div_sub_sum
        if div_sum == 0:
            print("for", i, "div_sum is zero")
            sum_samples += 0
        else:
            sum_samples += abs(sum_measurements / div_sum)
        monotonicity_single = sum_samples / len(X_single)-1
    return monotonicity_single


def Mo(X):
    """
    This function calculates the monotonicity value for a set of HIs.
    Input 'X':  matrix of all extracted HIs (m rows x n columns),
    where one row represents one HI and columns represent timesteps
    Output 'monotonicity': scalar, monotonicity value for set of HIs,
    ranges from 0 to 1
    """
    sum_monotonicities = 0
    for i in range(len(X)):
        monotonicity_i = Mo_single(X[i, :])
        sum_monotonicities += monotonicity_i
    monotonicity = sum_monotonicities / np.shape(X)[0]
    return monotonicity


def fitness(X, Mo_a=1, Tr_b=1, Pr_c=1):
    """
    This function calculates the fitness value for a set of HIs.
    Input 'X':  matrix of all extracted HIs (m rows x n columns),
    where one row represents one HI and columns represent timesteps
    Inputs 'Mo_a', 'Tr_b', 'Pr_c': weights of Mo, Tr and Pr in the
    fitness formula (integers), with default value 1
    Output 'ftn': scalar, fitness value for set of HIs,
    ranges from 0 to 3 (assuming weights are equal to the default value)
    Output 'error': scalar, measure of error with respect to max. achievable
    fitness, equal to sum of weights divided by ftn
    """
    monotonicity = Mo(X)
    trendability = Tr(X)
    prognosability = Pr(X)

    ftn = Mo_a * monotonicity + Tr_b * trendability + Pr_c * prognosability
    error = (Mo_a + Tr_b + Pr_c) / ftn

    return ftn, error
