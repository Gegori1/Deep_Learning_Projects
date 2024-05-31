@ echo off
IF EXIST venv (
    echo Virtual environment already exists.
) ELSE (
    echo Creating virtual environment...
    python -m venv venv
)
@ echo activate virtual environment
call venv\Scripts\activate.bat
@ echo install dependencies
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r sdk\requirements.txt
@ echo install jupyter kernel
venv\Scripts\python.exe -m pip install ipykernel -U
@ echo install local package
venv\Scripts\python.exe -m pip install -e sdk
@ echo deactivate virtual environment
deactivate