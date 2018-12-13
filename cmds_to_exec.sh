. activate python36
mv /home/cleptes/Programming/Python/genre_classification_data/splits_32k_1024fft /home/cleptes/Programming/Python/genre_classification_data/splits
python dataset_tools.py
python nn_tweaker.py "32k_1024fft_padding" 2>&1 | tee nn_tweaker_32k_1024fft.log
mv /home/cleptes/Programming/Python/genre_classification_data/splits /home/cleptes/Programming/Python/genre_classification_data/splits_32k_1024fft
mv /home/cleptes/Programming/Python/genre_classification_data/splits_16k_512fft_nopadding /home/cleptes/Programming/Python/genre_classification_data/splits
python dataset_tools.py
python nn_tweaker.py "16k_512fft_nopadding" 2>&1 | tee nn_tweaker_16k_512fft_nopadding.log
mv /home/cleptes/Programming/Python/genre_classification_data/splits /home/cleptes/Programming/Python/genre_classification_data/splits_16k_512fft_nopadding
