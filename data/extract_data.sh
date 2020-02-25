# DFDC Preview dataset
unzip deepfake_challenge_v01a.zip
mv fb_dfd_release_0.1_final dfdc_small

# DFDC Large dataset
unzip dfdc_train_full.zip
cd dfdc_train_full/
unzip *.zip
rm *.zip
cd ..
mv dfdc_train_full dfdc_large

# Process both datasets
cd dfdc_small/
python parse.py
cd ..

cd dfdc_large/
python parse.py
cd ..
