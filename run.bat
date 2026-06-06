@echo off
echo.
echo  Kutubxonalar o'rnatilmoqda...
py -m pip install streamlit plotly pandas numpy scipy -q
echo.
echo  Ilova ishga tushirilmoqda...
py -m streamlit run app.py
pause
