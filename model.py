"""
Climate Resilient Transportation System - Snowy Road Condition Detection
==========================================================================
Equations sourced from:
 [1] Yang & Lei (2024) - "Classification of Complex Snow and Ice Cover on Highway
     Pavement Based on Image-Meteorology-Temperature Fusion" - IEEE Sensors Journal
 [2] Vassilev (2024) - "Road Surface Characterization Using a 77-81 GHz Polarimetric Radar"
     - IEEE TITS
 [3] Ojala & Seppänen (2025) - "Lightweight Regression Model with Prediction Interval
     Estimation for Computer Vision-Based Winter Road Surface Condition Monitoring"
     - IEEE Transactions on Intelligent Vehicles
"""

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# SECTION 1: EQUATIONS FROM YANG & LEI [1]
# ============================================================

def eq1_grayscale_energy(G):
    """
    Eq. 1 [Yang & Lei]: Grayscale Co-occurrence Matrix Energy
    E = Σ_i Σ_j [G(i,j)]^2
    Measures textural uniformity of road surface image.
    """
    return np.sum(G ** 2)

def eq2_moment_of_inertia(G):
    """
    Eq. 2 [Yang & Lei]: Moment of Inertia (Contrast)
    I = Σ_{n=0}^{k-1} n^2 * [Σ_{|i-j|=n} G(i,j)]
    Measures local variations in the co-occurrence matrix.
    """
    k = G.shape[0]
    I = 0.0
    for n in range(k):
        diagonal_sum = sum(G[i, j] for i in range(k) for j in range(k) if abs(i - j) == n)
        I += (n ** 2) * diagonal_sum
    return I

def eq3_inverse_difference_moment(G):
    """
    Eq. 3 [Yang & Lei]: Inverse Difference Moment (Homogeneity)
    L = Σ_i Σ_j G(i,j) / [1 + (i-j)^2]
    Measures local homogeneity of the road surface texture.
    """
    k = G.shape[0]
    L = 0.0
    for i in range(k):
        for j in range(k):
            L += G[i, j] / (1 + (i - j) ** 2)
    return L

def eq4_entropy(G):
    """
    Eq. 4 [Yang & Lei]: Entropy of Grayscale Co-occurrence Matrix
    EN = -Σ_i Σ_j G(i,j) * log(G(i,j))
    Measures disorder/randomness of surface texture.
    """
    G_safe = np.where(G > 0, G, 1e-10)
    return -np.sum(G * np.log(G_safe))

def eq5_correlation(G):
    """
    Eq. 5 [Yang & Lei]: Correlation Feature
    C = Σ_i Σ_j [(i*j)*G(i,j) - mu_i*mu_j] / (sigma_i * sigma_j)
    Measures linear dependencies of gray-level values in the image.
    """
    k = G.shape[0]
    indices = np.arange(k)
    mu_i = np.sum(indices[:, None] * G)
    mu_j = np.sum(indices[None, :] * G)
    sigma_i = np.sqrt(np.sum(((indices[:, None] - mu_i) ** 2) * G))
    sigma_j = np.sqrt(np.sum(((indices[None, :] - mu_j) ** 2) * G))
    if sigma_i * sigma_j < 1e-10:
        return 0.0
    C = np.sum((indices[:, None] * indices[None, :]) * G - mu_i * mu_j) / (sigma_i * sigma_j)
    return C

def eq6_vip_score(X, y, n_components=2):
    """
    Eq. 6 [Yang & Lei]: Variable Importance in Projection (VIP)
    VIP_j = sqrt(k * Σ_h [r²(y,c_h) * w²_hj] / Σ_h r²(y,c_h))
    Selects most relevant input features for classification.
    """
    pls = PLSRegression(n_components=n_components)
    pls.fit(X, y)
    k = X.shape[1]
    SS = np.sum(pls.y_loadings_ ** 2 * np.sum(pls.x_scores_ ** 2, axis=0), axis=0)
    Vt = np.sum(pls.x_weights_ ** 2 * SS, axis=1)
    vip = np.sqrt(k * Vt / np.sum(SS))
    return vip

def eq7_average_precision(P_i_list):
    """
    Eq. 7 [Yang & Lei]: Average Precision (AP)
    AP = (1/n) * Σ_i P_i
    Evaluates classification performance across all cover types.
    """
    return np.mean(P_i_list)

def eq8_drap_3to4(AP_three, AP_four):
    """
    Eq. 8 [Yang & Lei]: Decline Rate of Average Precision (3→4 classes)
    DRAP_{3→4} = (AP_three - AP_four) / AP_three
    Measures performance degradation when adding a 4th cover class.
    """
    return (AP_three - AP_four) / AP_three

def eq9_drap_4to5(AP_four, AP_five):
    """
    Eq. 9 [Yang & Lei]: Decline Rate of Average Precision (4→5 classes)
    DRAP_{4→5} = (AP_four - AP_five) / AP_four
    Measures performance degradation when adding a 5th cover class.
    """
    return (AP_four - AP_five) / AP_four


# ============================================================
# SECTION 2: EQUATIONS FROM VASSILEV [2] - Polarimetric Radar
# ============================================================

def eq10_coherence_vector(S_HH, S_VV, S_HV):
    """
    Eq. 10 [Vassilev, Eq.3]: Coherence Vector k
    k = [S_HH + S_VV, S_HH - S_VV, 2*S_HV]^T
    Foundation of polarimetric decomposition for road surface classification.
    """
    return np.array([S_HH + S_VV, S_HH - S_VV, 2 * S_HV])

def eq11_coherence_matrix(M):
    """
    Eq. 11 [Vassilev, Eq.5]: Averaged Coherence Matrix T̂
    T̂ = M * M† / N
    Statistical matrix used to extract surface scattering properties.
    """
    N = M.shape[1]
    T = (M @ M.conj().T) / N
    return T

def eq12_eigenvalue_decomposition(T):
    """
    Eq. 12 [Vassilev, Eq.6]: Eigenvalue Decomposition of Coherence Matrix
    T̂ = Σ_i λ_i * [e_i * e_i†]
    Decomposes surface scattering into dominant mechanisms.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(T)
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvalues[idx], eigenvectors[:, idx]

def eq13_probability_weights(eigenvalues):
    """
    Eq. 13 [Vassilev, Eq.8]: Probability Weights P_i
    P_i = λ_i / (λ_1 + λ_2 + λ_3)
    Relative importance of each scattering mechanism.
    """
    total = np.sum(eigenvalues)
    return eigenvalues / (total + 1e-10)

def eq14_target_entropy(P_i):
    """
    Eq. 14 [Vassilev, Eq.7]: Polarimetric Target Entropy H
    H = -Σ_i P_i * log_3(P_i)
    H=0: single scattering; H=1: random scattering.
    Wet surface → lower H; Ice → higher H.
    """
    P_safe = np.where(P_i > 1e-10, P_i, 1e-10)
    return -np.sum(P_i * np.log(P_safe) / np.log(3))

def eq15_auxiliary_angle(P_i, eigenvectors):
    """
    Eq. 15 [Vassilev, Eq.9]: Mean Polarimetric Scattering Angle α
    α = Σ_i P_i * arccos(|e_i1|)
    α≈0°: surface scattering; α≈45°: volume; α≈90°: double bounce.
    """
    alpha = sum(P_i[i] * np.arccos(np.abs(eigenvectors[0, i]))
                for i in range(len(P_i)))
    return np.degrees(alpha)


# ============================================================
# SECTION 3: EQUATIONS FROM OJALA & SEPPÄNEN [3] - SIWNet
# ============================================================

def eq16_truncated_normal_pdf(x, mu, sigma, a=0.0, b=1.0):
    """
    Eq. 16 [Ojala & Seppänen, Eq.1]: Truncated Normal PDF
    p(μ,σ,a,b;x) = φ((x-μ)/σ) / [Φ((b-μ)/σ) - Φ((a-μ)/σ)]
    Models friction factor uncertainty bounded to [0,1].
    """
    from scipy.stats import truncnorm
    a_std = (a - mu) / sigma
    b_std = (b - mu) / sigma
    return truncnorm.pdf(x, a_std, b_std, loc=mu, scale=sigma)

def eq17_normal_pdf(x, mu, sigma):
    """
    Eq. 17 [Ojala & Seppänen, Eq.2]: Underlying Normal Distribution PDF φ
    φ((x-μ)/σ) = (1/σ√2π) * exp(-(x-μ)²/2σ²)
    Base distribution for the truncated normal friction model.
    """
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def eq18_normal_cdf(b, mu, sigma):
    """
    Eq. 18 [Ojala & Seppänen, Eq.3]: Normal CDF Φ
    Φ((b-μ)/σ) = (1/2)[1 + erf((b-μ)/(σ√2π))]
    Used for computing truncation bounds of friction distribution.
    """
    from scipy.special import erf
    return 0.5 * (1 + erf((b - mu) / (sigma * np.sqrt(2 * np.pi))))

def eq19_neg_log_likelihood(mu, sigma, x, a=0.0, b=1.0):
    """
    Eq. 19 [Ojala & Seppänen, Eq.4]: Negative Log-Likelihood Loss
    -ln p = ln σ + (μ-x)²/2σ² + ln[erf(...) - erf(...)]
    Training objective for the prediction interval head of SIWNet.
    """
    from scipy.special import erf
    sigma = max(sigma, 1e-4)
    term1 = np.log(sigma)
    term2 = (mu - x) ** 2 / (2 * sigma ** 2)
    erf_b = erf((mu - b) / (sigma * np.sqrt(2)))
    erf_a = erf((mu - a) / (sigma * np.sqrt(2)))
    term3 = np.log(np.abs(erf_b - erf_a) + 1e-10)
    return term1 + term2 + term3

def eq20_batch_loss(f_hat, sigma_hat, f_true, a=0.0, b=1.0):
    """
    Eq. 20 [Ojala & Seppänen, Eq.5]: Batch Training Loss L
    L = Σ_i -ln p(f̂_i, σ̂_i, a, b; f_i)
    Aggregated loss for the SIWNet prediction interval head.
    """
    return sum(eq19_neg_log_likelihood(f_hat[i], sigma_hat[i], f_true[i], a, b)
               for i in range(len(f_hat)))


# ============================================================
# SECTION 4: DERIVED / EXTENDED EQUATIONS FOR CLIMATE RESILIENCE
# ============================================================

def eq21_friction_factor_normalized(grip_factor, g_min=0.09, g_max=0.82):
    """
    Eq. 21 [Ojala & Seppänen, Section III-A]: Friction Factor Normalization
    f = (grip_factor - g_min) / (g_max - g_min)
    Normalizes optical sensor grip factor to [0, 1] friction factor f.
    """
    return (grip_factor - g_min) / (g_max - g_min)

def eq22_prediction_interval(mu, sigma, confidence=0.90):
    """
    Eq. 22 [Ojala & Seppänen, Section III-D]: Prediction Interval Construction
    PI = [F^{-1}((1-c)/2), F^{-1}((1+c)/2)] for truncated normal
    Constructs the 90% prediction interval around friction estimate.
    """
    from scipy.stats import truncnorm
    a_std = (0 - mu) / sigma
    b_std = (1 - mu) / sigma
    lower = truncnorm.ppf((1 - confidence) / 2, a_std, b_std, loc=mu, scale=sigma)
    upper = truncnorm.ppf((1 + confidence) / 2, a_std, b_std, loc=mu, scale=sigma)
    return lower, upper

def eq23_interval_score(y_true, lower, upper, alpha=0.10):
    """
    Eq. 23 [Ojala & Seppänen, Section III-D]: Interval Score (IS)
    IS = (u-l) + (2/α)(l-y)·1[y<l] + (2/α)(y-u)·1[y>u]
    Evaluates quality of prediction intervals; lower is better.
    """
    width = upper - lower
    penalty_low = np.where(y_true < lower, (2 / alpha) * (lower - y_true), 0)
    penalty_high = np.where(y_true > upper, (2 / alpha) * (y_true - upper), 0)
    return np.mean(width + penalty_low + penalty_high)

def eq24_crps(f_hat, sigma_hat, f_true):
    """
    Eq. 24 [Ojala & Seppänen, Section III-D]: Continuous Ranked Probability Score
    CRPS = E|X - y| - (1/2)E|X - X'|  ≈ MAE for point estimates
    Evaluates the full predictive distribution quality.
    """
    from scipy.stats import norm
    crps_vals = []
    for i in range(len(f_true)):
        mu, sigma = f_hat[i], max(sigma_hat[i], 1e-4)
        y = f_true[i]
        z = (y - mu) / sigma
        crps = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
        crps_vals.append(abs(crps))
    return np.mean(crps_vals)

def eq25_mae(y_true, y_pred):
    """
    Eq. 25 [Ojala & Seppänen, Table II]: Mean Absolute Error
    MAE = (1/n) Σ_i |f_i - f̂_i|
    Primary accuracy metric for friction factor regression.
    """
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def eq26_rmse(y_true, y_pred):
    """
    Eq. 26 [Ojala & Seppänen, Table II]: Root Mean Square Error
    RMSE = sqrt((1/n) Σ_i (f_i - f̂_i)²)
    Secondary accuracy metric penalizing large errors.
    """
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

def eq27_road_surface_scattering_model(entropy, alpha_angle):
    """
    Eq. 27 [Vassilev, Section VI]: Combined Surface Classification Score
    S = w1*(1-H) + w2*(1 - α/90)  — higher score → dryer surface
    Integrates entropy and alpha angle for road surface type estimation.
    """
    w1, w2 = 0.6, 0.4
    return w1 * (1 - entropy) + w2 * (1 - alpha_angle / 90.0)


# ============================================================
# SIMULATION: Generate synthetic road condition data
# ============================================================

def generate_road_condition_data(n_samples=600, seed=42):
    """
    Generates synthetic dataset representing 5 road surface cover types:
    Dry (DR), Fresh Snow (FS), Transparent Ice (TI), 
    Granular Snow (GS), Mixed Ice (MI)
    Based on experimental setup in Yang & Lei [1].
    """
    np.random.seed(seed)
    classes = ['Dry', 'Fresh Snow', 'Transparent Ice', 'Granular Snow', 'Mixed Ice']
    n_per_class = n_samples // len(classes)

    records = []
    for cls_idx, cls_name in enumerate(classes):
        for _ in range(n_per_class):
            # Image features (color moments + texture) - 33 features total
            if cls_name == 'Dry':
                color = np.random.normal(0.40, 0.05, 18)
                texture = np.random.normal(0.60, 0.04, 15)
                temp = np.random.normal(5, 3)
                friction = np.random.normal(0.85, 0.05)
                entropy = np.random.normal(0.74, 0.04)
                alpha = np.random.normal(38, 3.2)
            elif cls_name == 'Fresh Snow':
                color = np.random.normal(0.78, 0.06, 18)
                texture = np.random.normal(0.35, 0.05, 15)
                temp = np.random.normal(-5, 3)
                friction = np.random.normal(0.38, 0.07)
                entropy = np.random.normal(0.72, 0.04)
                alpha = np.random.normal(42, 3.5)
            elif cls_name == 'Transparent Ice':
                color = np.random.normal(0.52, 0.04, 18)
                texture = np.random.normal(0.67, 0.03, 15)
                temp = np.random.normal(-8, 4)
                friction = np.random.normal(0.19, 0.04)
                entropy = np.random.normal(0.78, 0.03)
                alpha = np.random.normal(23, 2.4)
            elif cls_name == 'Granular Snow':
                color = np.random.normal(0.67, 0.05, 18)
                texture = np.random.normal(0.46, 0.05, 15)
                temp = np.random.normal(-3, 2)
                friction = np.random.normal(0.50, 0.06)
                entropy = np.random.normal(0.71, 0.036)
                alpha = np.random.normal(41, 3.1)
            else:  # Mixed Ice
                color = np.random.normal(0.60, 0.05, 18)
                texture = np.random.normal(0.56, 0.04, 15)
                temp = np.random.normal(-6, 3)
                friction = np.random.normal(0.25, 0.05)
                entropy = np.random.normal(0.76, 0.035)
                alpha = np.random.normal(27, 2.37)

            # Meteorological features: UV, illuminance, wind_speed, humidity, air_temp, body_temp
            meteo = np.array([
                np.random.normal(0.3, 0.1),   # UV index
                np.random.normal(500, 200),    # illuminance (lux)
                np.random.normal(3, 1.5),      # wind speed (m/s)
                np.random.normal(75, 15),      # humidity (%)
                temp + np.random.normal(0, 1), # air temp
                temp + np.random.normal(-2, 1) # body/surface temp
            ])

            records.append({
                'class': cls_name,
                'class_id': cls_idx,
                'friction': np.clip(friction, 0.09, 1.0),
                'entropy': np.clip(entropy, 0, 1),
                'alpha_angle': np.clip(alpha, 0, 90),
                'temperature': temp,
                **{f'color_{i}': color[i % len(color)] for i in range(18)},
                **{f'texture_{i}': texture[i % len(texture)] for i in range(15)},
                **{f'meteo_{i}': meteo[i] for i in range(6)},
            })

    df = pd.DataFrame(records)
    df['friction'] = df['friction'].clip(0, 1)
    return df


def run_classification_experiments(df):
    """
    Runs the multi-method classification experiments as described in Yang & Lei [1].
    Methods: MM, TM, IM, MTFM, IMFM, ITFM, IMTFM
    Classification scenarios: 3-class, 4-class (2 variants), 5-class
    """
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder

    results = {}
    scenarios = {
        '3-class': ['Dry', 'Fresh Snow', 'Transparent Ice'],
        '4-class-I': ['Dry', 'Fresh Snow', 'Transparent Ice', 'Granular Snow'],
        '4-class-II': ['Dry', 'Fresh Snow', 'Transparent Ice', 'Mixed Ice'],
        '5-class': ['Dry', 'Fresh Snow', 'Transparent Ice', 'Granular Snow', 'Mixed Ice'],
    }

    feature_sets = {
        'MM': [f'meteo_{i}' for i in range(6)],
        'TM': ['temperature'],
        'IM': [f'color_{i}' for i in range(18)] + [f'texture_{i}' for i in range(15)],
        'MTFM': [f'meteo_{i}' for i in range(6)] + ['temperature'],
        'IMFM': [f'color_{i}' for i in range(18)] + [f'texture_{i}' for i in range(15)] + [f'meteo_{i}' for i in range(6)],
        'ITFM': [f'color_{i}' for i in range(18)] + [f'texture_{i}' for i in range(15)] + ['temperature'],
        'IMTFM': [f'color_{i}' for i in range(18)] + [f'texture_{i}' for i in range(15)] + [f'meteo_{i}' for i in range(6)] + ['temperature'],
    }

    le = LabelEncoder()
    for scenario, classes in scenarios.items():
        df_s = df[df['class'].isin(classes)].copy()
        df_s['label'] = le.fit_transform(df_s['class'])
        results[scenario] = {}

        for method, feats in feature_sets.items():
            available = [f for f in feats if f in df_s.columns]
            X = df_s[available].values
            y = df_s['label'].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            n_comp = min(2, X_train.shape[1], X_train.shape[0] - 1)
            pls = PLSRegression(n_components=n_comp)
            pls.fit(X_train, y_train)
            y_pred = np.round(np.clip(pls.predict(X_test).flatten(), 0, len(classes) - 1)).astype(int)

            per_class_acc = []
            for c in range(len(classes)):
                mask = y_test == c
                if mask.sum() > 0:
                    per_class_acc.append(accuracy_score(y_test[mask], y_pred[mask]))

            ap = eq7_average_precision(per_class_acc)
            results[scenario][method] = {
                'AP': ap,
                'per_class': per_class_acc,
                'n_classes': len(classes)
            }

    return results


if __name__ == '__main__':
    print("Generating road condition dataset...")
    df = generate_road_condition_data(n_samples=600)
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['class'].value_counts()}")

    print("\nRunning classification experiments...")
    results = run_classification_experiments(df)
    for scenario, methods in results.items():
        print(f"\n{scenario}:")
        for method, metrics in methods.items():
            print(f"  {method}: AP={metrics['AP']:.3f}")

    df.to_csv('road_condition_data.csv', index=False)
    print("\nData saved!")
