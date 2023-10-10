for i in `seq 1 100`
do
    echo $i
    time python3 D2A2.py $i
    time python3 D2A3.py $i
    time python3 D2A4.py $i
    time python3 D2A5.py $i
    time python3 D2A6.py $i
    time python3 D3A4.py $i
    time python3 D3A6.py $i
    time python3 D4A8.py $i
done