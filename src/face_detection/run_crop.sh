for i in {0..5}
do
    python crop_all.py --process_id $i --total_processes 50 --workers 1  --data_dir ../../data/ --out_dir dfdc_bursted_final --exp_name crop_extract_$i &
done
wait
echo "Done!"

