for i in {0..49}
do
    python parse.py $i &
done
wait
python combine.py
echo "Done!"
