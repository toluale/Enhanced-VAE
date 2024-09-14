def main():
    configure_gpu()
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        file_paths = {
            "test": '/home/tale2@ad.umbc.edu/Dataset/wadi/test.csv',
            "train": '/home/tale2@ad.umbc.edu/Dataset/wadi/train.csv',
            "label": '/home/tale2@ad.umbc.edu/Dataset/wadi/label.csv'
        }

        train = load_data(file_paths['train'])
        clustered_dataframes, cluster_labels = cluster_data(train, N_CLUSTERS)

        x = [create_sequences(clustered_dataframes[i].values, T) for i in range(N_CLUSTERS)]
        
        # Verify the shape of each sequence
        for i, seq in enumerate(x):
            print(f"Cluster {i}: Sequence shape {seq.shape}")
            
            assert seq.ndim == 3, f"Sequence for cluster {i} does not have 3 dimensions"

        auto = vae_model(x)

        total_features = sum([df.shape[2] for df in x])
        output_data = np.concatenate([df.reshape(df.shape[0], df.shape[1], df.shape[2]) for df in x], axis=-1)

        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10, verbose=1, mode='auto', restore_best_weights=True)
        history = auto.fit(x, output_data, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, callbacks=[monitor], shuffle=False).history

        train_pred = auto.predict(x)

        keys = list(clustered_dataframes.keys())
        actual_train = clustered_dataframes[keys[0]]  
        for key in keys[1:]:  
            actual_train = actual_train.join(clustered_dataframes[key], how='outer')  

        # Training Data Processing and MSE Calculation
        #train_pred = sequences_to_original(train_pred, T)
        #actual_train = clustered_dataframes[0].join(clustered_dataframes[1:])
        #actual_train2 = actual_train.values
        #train_mse = mean_squared_error(actual_train2, train_pred)
        #print(f"Mean Squared Error (Train): {train_mse}")

        # Test Data Processing and MSE Calculation
        test_data = load_data(file_paths['test'])  # Load the test data
        X_test = process_test_data(test_data, clustered_dataframes, T)

        #X_test, test_data = process_data(file_paths['test'], cluster_labels, T)
        test_pred = auto.predict(X_test)
        test_pred = sequences_to_original(test_pred, T)
        actual_test = test_data[actual_train.columns].values
        #actual_test = test_data.values

        # Ensure the test_pred and actual_test have the same shape before calculating MSE
        if test_pred.shape != actual_test.shape:
            min_shape = min(test_pred.shape[0], actual_test.shape[0])
            test_pred = test_pred[:min_shape, :]
            actual_test = actual_test[:min_shape, :]
        test_mse = mean_squared_error(actual_test, test_pred)
        print(f"Mean Squared Error (Test): {test_mse}")
        print(f"test_pred: {test_pred.shape}")
        print(f"actual_test: {actual_test.shape}")

        # Anomaly Detection
        scored = np.mean(np.abs(test_pred - actual_test), axis=1)
        print(f"scored: {scored.shape}")
        
        predicted_labels = dynamic_thresholding(scored)
        print(f"predicted_labels: {predicted_labels.shape}")

        true_labels = load_data(file_paths['label']).values
        evaluate_model(true_labels, predicted_labels)
                

if __name__ == "__main__":
    main()
