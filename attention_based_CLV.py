import tensorflow as tf
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.metrics import mean_squared_error, confusion_matrix, precision_score, recall_score, f1_score

from tensorflow.keras.layers import (Input, LSTM, RepeatVector, TimeDistributed, Dense,
                                    Concatenate, Lambda, Dropout, MultiHeadAttention, BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import tensorflow.keras.backend as K

# constants
T = 15
SEED = 50
N_CLUSTERS = 10
LEARNING_RATE = 0.0001
EPOCHS = 20
BATCH_SIZE = 512
window_size = 85000

def configure_gpu():
    """Configure TensorFlow to use specified GPUs."""
    gpus = tf.config.experimental.list_physical_devices("GPU")
    #tf.config.experimental.set_visible_devices([gpus[1], gpus[2]], "GPU")

def load_data(filepath):
    """Load data from CSV files."""
    return pd.read_csv(filepath, index_col=0)

def cluster_data(data, n_clusters):
    """
    Cluster data based on correlation using Hierarchical Clustering
    and return clustered dataframes.
    """
    corr_mat = data.corr()
    dist_mat = pairwise_distances(corr_mat, metric='euclidean')  # Distance matrix from correlation matrix

    # Perform hierarchical clustering (Agglomerative)
    clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
    cluster_labels = clustering.fit_predict(dist_mat)

    clustered_dataframes = {}
    for label in np.unique(cluster_labels):
        columns_in_cluster = [col_name for col_name, col_cluster in zip(data.columns, cluster_labels) if col_cluster == label]
        clustered_dataframes[label] = data[columns_in_cluster]

    return clustered_dataframes, cluster_labels
                              
def create_sequences(data, seq_length):
    xs = []
    #ys = []

    for i in range(len(data)-seq_length):
        x = data[i:(i+seq_length)]
        #y = data[i+seq_length]
        xs.append(x)
        #ys.append(y)

    return np.array(xs) #, np.array(ys)

def sequences_to_original(X, seq_length):
    original_shape = np.zeros((X.shape[0] + seq_length, ) + X.shape[2:])
    for i in range(X.shape[0]):
        original_shape[i] = X[i][0]
    return original_shape

def get_clusters(clustered_dataframes, index):
    return clustered_dataframes[index]

def process_data(file_path, cluster_labels, T):
    data = load_data(file_path)

    X = []
    for cluster_label in np.unique(cluster_labels):  # Iterate over unique cluster labels
        cluster_data = data[cluster_labels == cluster_label]  # Filter rows belonging to the current cluster
        sequences = create_sequences(cluster_data.values, T)  # Create sequences from filtered data
        X.append(sequences)

    return X, data

def process_test_data(test_data, train_clustered_dataframes, T):
    X_test = []
    for cluster_label, clustered_train_df in train_clustered_dataframes.items():
        cluster_columns = clustered_train_df.columns
        if not all(col in test_data.columns for col in cluster_columns):
            raise ValueError(f"Some columns in the train data cluster {cluster_label} are missing in the test data.")
        cluster_test_data = test_data[cluster_columns]
        test_sequences = create_sequences(cluster_test_data.values, T)
        X_test.append(test_sequences)
    return X_test

""""

def dynamic_thresholding(scored, window_size=window_size, threshold_percentile=90):
    anomaly_results = []
    # Handle first and last points with dedicated function
#    def handle_edge_points(points):
        threshold = np.percentile(points, threshold_percentile)
        peaks_over_threshold = points[points > threshold]
        shape, location, scale = stats.genpareto.fit(peaks_over_threshold)
        probs = 1 - stats.genpareto.cdf(points, shape, location, scale)
        return [1 if prob < 0.50 else 0 for prob in probs]

    anomaly_results.extend(handle_edge_points(scored[:window_size]))

#    for i in range(window_size, len(scored) - window_size):
        window_data = scored[i-window_size:i+window_size+1]
        threshold = np.percentile(window_data, threshold_percentile)
        peaks_over_threshold = window_data[window_data > threshold]
        shape, location, scale = stats.genpareto.fit(peaks_over_threshold)
        probs = 1 - stats.genpareto.cdf(window_data, shape, location, scale)
        anomaly_results.append(1 if probs[window_size] < 0.50 else 0)

    anomaly_results.extend(handle_edge_points(scored[-window_size:]))

    return pd.DataFrame(anomaly_results, columns=['label'])
    """

def dynamic_thresholding(scored, window_size=window_size, threshold_percentile=90):
    anomaly_results = []
    
    def calculate_anomalies(points):
        """Helper function to calculate anomalies for a given set of points."""
        threshold = np.percentile(points, threshold_percentile)
        peaks_over_threshold = points[points > threshold]
        if len(peaks_over_threshold) > 0:
            shape, location, scale = stats.genpareto.fit(peaks_over_threshold)
            probs = 1 - stats.genpareto.cdf(points, shape, location, scale)
            return [1 if prob < 0.50 else 0 for prob in probs]
        else:
            # If no peaks over threshold, return all non-anomalies
            return [0] * len(points)
    scored = np.asarray(scored)
    
    # Handle the first 'window_size' points
    first_points = scored[:window_size]
    anomaly_results.extend(calculate_anomalies(first_points))
    
    # Iterate over the data with a moving window for the middle points
    for i in range(window_size, len(scored) - window_size):
        window_data = scored[i-window_size:i+window_size+1]
        is_anomaly = calculate_anomalies(window_data)[window_size]
        anomaly_results.append(is_anomaly)
    
    # Handle the last 'window_size' points
    last_points = scored[-window_size:]
    anomaly_results.extend(calculate_anomalies(last_points))
    
    return pd.DataFrame(anomaly_results, columns=['label'])

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def vae_model(X_list, kl_weight=1):
    """Create a VAE model."""
    inputs = []
    encoders = []
    total_features = sum([X.shape[2] for X in X_list])

    for X in X_list:
        inp = Input(shape=(X.shape[1], X.shape[2]))
        enc = LSTM(18, activation='relu', return_sequences=True, kernel_initializer=GlorotUniform())(inp)
        enc = LSTM(6, activation='relu', return_sequences=True, kernel_initializer=GlorotUniform())(enc)
        attention_enc = MultiHeadAttention(num_heads=2, key_dim=4)(enc, enc)
        attention_enc = Dropout(0.5)(attention_enc)
        attention_enc = BatchNormalization()(attention_enc)
        enc_att = LSTM(6, activation='relu', return_sequences=False, kernel_initializer=GlorotUniform())(attention_enc)
        z_mean = Dense(5)(enc_att)
        z_log_var = Dense(5)(enc_att)
        z = Lambda(sampling, output_shape=(5,))([z_mean, z_log_var])
        inputs.append(inp)
        encoders.append(z)

    concat = Concatenate(axis=-1)(encoders)
    dec = RepeatVector(max([X.shape[1] for X in X_list]))(concat)
    dec = LSTM(48, activation='relu', return_sequences=True, kernel_initializer=GlorotUniform())(dec)
    dec = LSTM(64, activation='relu', return_sequences=True, kernel_initializer=GlorotUniform())(dec)
    attention_dec = MultiHeadAttention(num_heads=2, key_dim=4)(dec, dec)
    attention_dec = Dropout(0.5)(attention_dec)
    attention_dec = BatchNormalization()(attention_dec)
    output = TimeDistributed(Dense(total_features))(attention_dec)

    vae = Model(inputs=inputs, outputs=output)

    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(K.clip(z_log_var, -10, 10)), axis=-1)
    vae.add_loss(kl_weight * K.mean(kl_loss))
    vae.compile(optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm =1.0), loss='mse')

    return vae

def evaluate_model(true_labels, predicted_labels):
    print("Confusion Matrix:", confusion_matrix(true_labels, predicted_labels))
    print("Precision:", precision_score(true_labels, predicted_labels))
    print("Recall:", recall_score(true_labels, predicted_labels))
    print("F1 Score:", f1_score(true_labels, predicted_labels))
