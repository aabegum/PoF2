@echo off
REM ============================================================================
REM POF2 PIPELINE - FULL EXECUTION WITH LOGGING
REM Turkish EDAS Equipment Failure Prediction Pipeline
REM ============================================================================

setlocal enabledelayedexpansion

REM Create logs directory with timestamp
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c-%%a-%%b)
for /f "tokens=1-2 delims=/:" %%a in ('time /t') do (set mytime=%%a%%b)
set timestamp=%mydate%_%mytime%
set LOG_DIR=logs\run_%timestamp%

echo ============================================================================
echo                    POF2 PIPELINE EXECUTION
echo ============================================================================
echo.
echo Creating log directory: %LOG_DIR%
mkdir "%LOG_DIR%" 2>nul

REM Master log file (all outputs combined)
set MASTER_LOG=%LOG_DIR%\pipeline_master.log

echo Pipeline started at %date% %time% > "%MASTER_LOG%"
echo ============================================================================ >> "%MASTER_LOG%"
echo. >> "%MASTER_LOG%"

REM Start time
set start_time=%time%

REM ============================================================================
REM STEP 1: DATA PROFILING
REM ============================================================================
echo.
echo [STEP 1/9] Running Data Profiling...
echo [STEP 1/9] Running Data Profiling... >> "%MASTER_LOG%"
python 01_data_profiling.py > "%LOG_DIR%\01_data_profiling.log" 2>&1
if errorlevel 1 (
    echo ERROR: Step 1 failed! Check %LOG_DIR%\01_data_profiling.log
    echo ERROR: Step 1 failed! >> "%MASTER_LOG%"
    goto :error
)
type "%LOG_DIR%\01_data_profiling.log" >> "%MASTER_LOG%"
echo [STEP 1/9] ✓ Completed

REM ============================================================================
REM STEP 2: DATA TRANSFORMATION
REM ============================================================================
echo.
echo [STEP 2/9] Running Data Transformation...
echo [STEP 2/9] Running Data Transformation... >> "%MASTER_LOG%"
python 02_data_transformation.py > "%LOG_DIR%\02_data_transformation.log" 2>&1
if errorlevel 1 (
    echo ERROR: Step 2 failed! Check %LOG_DIR%\02_data_transformation.log
    echo ERROR: Step 2 failed! >> "%MASTER_LOG%"
    goto :error
)
type "%LOG_DIR%\02_data_transformation.log" >> "%MASTER_LOG%"
echo [STEP 2/9] ✓ Completed

REM ============================================================================
REM STEP 3: FEATURE ENGINEERING
REM ============================================================================
echo.
echo [STEP 3/9] Running Feature Engineering...
echo [STEP 3/9] Running Feature Engineering... >> "%MASTER_LOG%"
python 03_feature_engineering.py > "%LOG_DIR%\03_feature_engineering.log" 2>&1
if errorlevel 1 (
    echo ERROR: Step 3 failed! Check %LOG_DIR%\03_feature_engineering.log
    echo ERROR: Step 3 failed! >> "%MASTER_LOG%"
    goto :error
)
type "%LOG_DIR%\03_feature_engineering.log" >> "%MASTER_LOG%"
echo [STEP 3/9] ✓ Completed

REM ============================================================================
REM STEP 4: EXPLORATORY DATA ANALYSIS
REM ============================================================================
echo.
echo [STEP 4/9] Running Exploratory Data Analysis...
echo [STEP 4/9] Running Exploratory Data Analysis... >> "%MASTER_LOG%"
python 04_eda.py > "%LOG_DIR%\04_eda.log" 2>&1
if errorlevel 1 (
    echo ERROR: Step 4 failed! Check %LOG_DIR%\04_eda.log
    echo ERROR: Step 4 failed! >> "%MASTER_LOG%"
    goto :error
)
type "%LOG_DIR%\04_eda.log" >> "%MASTER_LOG%"
echo [STEP 4/9] ✓ Completed

REM ============================================================================
REM STEP 5: FEATURE SELECTION
REM ============================================================================
echo.
echo [STEP 5/9] Running Feature Selection...
echo [STEP 5/9] Running Feature Selection... >> "%MASTER_LOG%"
python 05_feature_selection.py > "%LOG_DIR%\05_feature_selection.log" 2>&1
if errorlevel 1 (
    echo ERROR: Step 5 failed! Check %LOG_DIR%\05_feature_selection.log
    echo ERROR: Step 5 failed! >> "%MASTER_LOG%"
    goto :error
)
type "%LOG_DIR%\05_feature_selection.log" >> "%MASTER_LOG%"
echo [STEP 5/9] ✓ Completed

REM ============================================================================
REM STEP 6: REMOVE LEAKY FEATURES
REM ============================================================================
echo.
echo [STEP 6/9] Running Remove Leaky Features...
echo [STEP 6/9] Running Remove Leaky Features... >> "%MASTER_LOG%"
python 05b_remove_leaky_features.py > "%LOG_DIR%\05b_remove_leaky_features.log" 2>&1
if errorlevel 1 (
    echo ERROR: Step 6 failed! Check %LOG_DIR%\05b_remove_leaky_features.log
    echo ERROR: Step 6 failed! >> "%MASTER_LOG%"
    goto :error
)
type "%LOG_DIR%\05b_remove_leaky_features.log" >> "%MASTER_LOG%"
echo [STEP 6/9] ✓ Completed

REM ============================================================================
REM STEP 7: MODEL TRAINING (CHRONIC REPEATER - MODEL 2)
REM ============================================================================
echo.
echo [STEP 7/9] Running Model Training (Chronic Repeater - Model 2)...
echo [STEP 7/9] Running Model Training (Chronic Repeater - Model 2)... >> "%MASTER_LOG%"
python 06_model_training.py > "%LOG_DIR%\06_model_training.log" 2>&1
if errorlevel 1 (
    echo ERROR: Step 7 failed! Check %LOG_DIR%\06_model_training.log
    echo ERROR: Step 7 failed! >> "%MASTER_LOG%"
    goto :error
)
type "%LOG_DIR%\06_model_training.log" >> "%MASTER_LOG%"
echo [STEP 7/9] ✓ Completed

REM ============================================================================
REM STEP 8: SURVIVAL ANALYSIS (TEMPORAL POF - MODEL 1)
REM ============================================================================
echo.
echo [STEP 8/9] Running Survival Analysis (Temporal PoF - Model 1)...
echo [STEP 8/9] Running Survival Analysis (Temporal PoF - Model 1)... >> "%MASTER_LOG%"
python 09_survival_analysis.py > "%LOG_DIR%\09_survival_analysis.log" 2>&1
if errorlevel 1 (
    echo ERROR: Step 8 failed! Check %LOG_DIR%\09_survival_analysis.log
    echo ERROR: Step 8 failed! >> "%MASTER_LOG%"
    goto :error
)
type "%LOG_DIR%\09_survival_analysis.log" >> "%MASTER_LOG%"
echo [STEP 8/9] ✓ Completed

REM ============================================================================
REM STEP 9: CONSEQUENCE OF FAILURE & RISK SCORING
REM ============================================================================
echo.
echo [STEP 9/9] Running Consequence of Failure and Risk Scoring...
echo [STEP 9/9] Running Consequence of Failure and Risk Scoring... >> "%MASTER_LOG%"
python 10_consequence_of_failure.py > "%LOG_DIR%\10_consequence_of_failure.log" 2>&1
if errorlevel 1 (
    echo ERROR: Step 9 failed! Check %LOG_DIR%\10_consequence_of_failure.log
    echo ERROR: Step 9 failed! >> "%MASTER_LOG%"
    goto :error
)
type "%LOG_DIR%\10_consequence_of_failure.log" >> "%MASTER_LOG%"
echo [STEP 9/9] ✓ Completed

REM ============================================================================
REM SUCCESS
REM ============================================================================
echo.
echo ============================================================================
echo                    PIPELINE COMPLETED SUCCESSFULLY
echo ============================================================================
echo.
echo Pipeline completed at %date% %time%
echo Start time: %start_time%
echo End time:   %time%
echo.
echo 📂 Log Files Location: %LOG_DIR%
echo    • Individual logs: %LOG_DIR%\*.log
echo    • Master log (all outputs): %LOG_DIR%\pipeline_master.log
echo.
echo 📊 Output Files Generated:
echo    • results\risk_assessment_*.csv
echo    • results\capex_priority_list.csv
echo    • outputs\eda\*.png
echo    • outputs\risk_analysis\*.png
echo.

echo. >> "%MASTER_LOG%"
echo ============================================================================ >> "%MASTER_LOG%"
echo Pipeline completed successfully at %date% %time% >> "%MASTER_LOG%"
echo ============================================================================ >> "%MASTER_LOG%"

goto :end

REM ============================================================================
REM ERROR HANDLER
REM ============================================================================
:error
echo.
echo ============================================================================
echo                    PIPELINE FAILED
echo ============================================================================
echo.
echo Pipeline failed at %date% %time%
echo Please check the log files in: %LOG_DIR%
echo.
echo. >> "%MASTER_LOG%"
echo ============================================================================ >> "%MASTER_LOG%"
echo Pipeline FAILED at %date% %time% >> "%MASTER_LOG%"
echo ============================================================================ >> "%MASTER_LOG%"
exit /b 1

:end
endlocal
