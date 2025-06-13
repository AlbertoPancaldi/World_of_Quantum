# World_of_Quantum repo

1. Clone the repo 
    git clone https://github.com/AlbertoPancaldi/World_of_Quantum.git

2. Create and activate a virtual environment
    python3 -m venv venv_WoQ
    source venv_WoQ/bin/activate

3. Install dependencies
    pip install --upgrade pip
    pip install -r requirements.txt

4. Create the .gitignore file
    cat << EOF > .gitignore
    # Ignore the virtual-environment folder
    venv_WoQ/

    # Python bytecode and cache
    __pycache__/
    *.py[cod]

    # Jupyter notebook checkpoints
    .ipynb_checkpoints/

    # Optional extras
    *.log
    *.tmp
    EOF

5. Stage your changes 
   git add .
   git commit -m "Briefly describe what you changed"
   git push origin main

6. Pulling Remote Changes
    git pull origin main

7. Sanity check
    git status