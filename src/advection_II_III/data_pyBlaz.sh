#download the data
#produce results in the appropraite folder using the scripts
git clone https://github.com/damtharvey/pyblaz.git
cd pyblaz 
pip install -e .

cd pyblaz 
cp ../../save_pyblaz_4_8.py .
cp ../../save_pyblaz_4_16.py .
cp ../../save_pyblaz_8_8.py .
cp ../../save_pyblaz_8_16.py .

python3 save_pyblaz_4_8.py
python3 save_pyblaz_4_16.py
python3 save_pyblaz_8_8.py
python3 save_pyblaz_8_16.py


#plotting the graph