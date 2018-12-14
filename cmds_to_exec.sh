. activate python36
#mv /home/cleptes/Programming/Python/genre_classification_data/splits_32k_1024fft /home/cleptes/Programming/Python/genre_classification_data/splits
#python dataset_tools.py
#python nn_tweaker.py "32k_1024fft_padding" 2>&1 | tee nn_tweaker_32k_1024fft.log
#mv /home/cleptes/Programming/Python/genre_classification_data/splits /home/cleptes/Programming/Python/genre_classification_data/splits_32k_1024fft
#mv /home/cleptes/Programming/Python/genre_classification_data/splits_16k_512fft_nopadding /home/cleptes/Programming/Python/genre_classification_data/splits
#python dataset_tools.py
#python nn_tweaker.py "16k_512fft_nopadding" 2>&1 | tee nn_tweaker_16k_512fft_nopadding.log
#mv /home/cleptes/Programming/Python/genre_classification_data/splits /home/cleptes/Programming/Python/genre_classification_data/splits_16k_512fft_nopadding
echo "`date`: started moving" > /home/cleptes/log_ml_run.log
mv /home/cleptes/Programming/Python/genre_classification_data/splits_32k_1024fft_nopadding /home/cleptes/Programming/Python/genre_classification_data/splits
echo "`date`: ended moving, starting dataset" >> /home/cleptes/log_ml_run.log

python dataset_tools.py
echo "`date`: ended dataset, starting learning" >> /home/cleptes/log_ml_run.log

python nn_tweaker.py "32k_1024fft_nopadding" 2>&1 | tee nn_tweaker_32k_1024fft_nopadding.log
echo "`date`: ended learning, starting moving" >> /home/cleptes/log_ml_run.log

mv /home/cleptes/Programming/Python/genre_classification_data/splits /home/cleptes/Programming/Python/genre_classification_data/splits_32k_1024fft_nopadding
echo "`date`: ended moving, starting mp3 to spectrogram" >> /home/cleptes/log_ml_run.log

cd /home/cleptes/Programming/Python/mp3_to_wav_to_spectogram
python parallel_mp3_to_spectrogram.py 2>&1 | tee parallel_mp3_to_spectrogram.log
echo "`date`: ended mp3 to spectrogram, starting slicing spectrograms" >> /home/cleptes/log_ml_run.log
python do_stuff_with_spectrogram_data.py 2>&1 | tee do_stuff_with_spectrogram_data.log
echo "`date`: ended" >> /home/cleptes/log_ml_run.log
