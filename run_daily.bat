@echo off
cd /d "d:\Statistical Arbitrage System"
echo [%date% %time%] Starting daily execution... >> data\execution\scheduler.log
python -m stat_arb.execution.run --live --update-data >> data\execution\scheduler.log 2>&1
echo [%date% %time%] Done. >> data\execution\scheduler.log
