@echo off
REM Runner Windows para la comparacion WO_base vs WO_shap sobre la instancia grande.
REM Llama al runner unificado runners/compare_base_vs_shap.py con --init-mode random.
REM
REM Uso:
REM   run.bat                       (1 corrida)
REM   set RUNS=30 ^&^& run.bat       (override numero de corridas via env)
REM   run.bat --runs 5               (override via argumento)
REM
REM Salidas: experiments\tmlap_grande\outputs\run_<RUNS>r\

setlocal EnableExtensions EnableDelayedExpansion
set THIS_DIR=%~dp0
set PROJECT_ROOT=%THIS_DIR%..\..

if "%RUNS%"==""        set RUNS=1
if "%AGENTS%"==""      set AGENTS=30
if "%ITERATIONS%"==""  set ITERATIONS=300
if "%SEED%"==""        set SEED=1234
if "%PROFILE%"==""     set PROFILE=soft
if "%SHAPLEY_STEPS%"=="" set SHAPLEY_STEPS=3

if not exist "%THIS_DIR%4.instancia_grande.txt" (
    echo ^>^> Generando 4.instancia_grande.txt ^(no existe^)...
    pushd "%THIS_DIR%"
    python generate_instance.py
    popd
)

set OUTPUT_DIR=%THIS_DIR%outputs\run_%RUNS%r_%PROFILE%
echo ^>^> WO_base vs WO_shap sobre tmlap:4.instancia_grande.txt
echo    runs=%RUNS% agents=%AGENTS% iterations=%ITERATIONS% seed=%SEED% profile=%PROFILE% shapley_steps=%SHAPLEY_STEPS%
echo    output=%OUTPUT_DIR%

pushd "%PROJECT_ROOT%"
python -m runners.compare_base_vs_shap ^
    --problem "tmlap:experiments\tmlap_grande\4.instancia_grande.txt" ^
    --runs %RUNS% ^
    --agents %AGENTS% ^
    --iterations %ITERATIONS% ^
    --seed %SEED% ^
    --profile %PROFILE% ^
    --shapley-steps %SHAPLEY_STEPS% ^
    --init-mode random ^
    --output "%OUTPUT_DIR%" ^
    %*
popd

echo ^>^> Listo. Resultados en %OUTPUT_DIR%
endlocal
