# Cluster_LSTM_VAE
This algorithm exploits the relationships between variables to improve the reconstruction performance of the variational autoencoder (VAE). A correlation score was used as the metric to group the features via a distance-based clustering method. The resulting clusters served as inputs for the Attention-Based VAE.

For anomaly detection, I employed KL-divergence and MSE loss as the anomaly score. Anomalous periods were identified using a dynamic thresholding technique applied to the anomaly scores, leveraging a sliding window mechanism.

The results of this experiment were validated on two benchmark datasets: WaDI and SMD. However, the primary goal of this model is to learn and infer trends and patterns of climate anomalies in the Arctic region. Specifically, it aims to analyze the intensification of extreme events, such as severe snow melt or temperature increases in the Arctic.

This work was presented (ORAL) at the IGARSS 2024 - IEEE International Geoscience and Remote Sensing Symposium in Athens, Greece. You can access the publication here: https://doi.org/10.1109/IGARSS53475.2024.10640794.



