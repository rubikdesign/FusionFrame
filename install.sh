#!/bin/bash
# Script pentru crearea unui mediu virtual cu dependențe compatibile pentru FusionFrame 2.0

echo "===== Configurare mediu pentru FusionFrame 2.0 ====="
echo "Creez mediul virtual..."

# Crearea mediului virtual
python -m venv fusionframe_env
source fusionframe_env/bin/activate

# Upgrade pip
pip install --upgrade pip

echo "Instalez dependențele..."
pip install -r requirements.txt

echo "Creez script de execuție..."
# Creăm scriptul de execuție
cat > run_fusionframe.sh << 'EOL'
#!/bin/bash
source fusionframe_env/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
python fusionframe/app.py "$@"
EOL

chmod +x run_fusionframe.sh

echo "===== Instalare completă! ====="
echo "Pentru a rula aplicația, folosește comanda: ./run_fusionframe.sh"
echo "NOTĂ: La prima rulare, modelele vor fi descărcate automat (poate dura câteva minute)."