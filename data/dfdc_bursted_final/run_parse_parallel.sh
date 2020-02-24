for i in {0..49}
do
    python parse.py $i &
done
wait
echo "Done!"
