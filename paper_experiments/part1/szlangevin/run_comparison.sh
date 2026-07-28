
# ACKLEY
# python3 comparison.py fd ackley --d 100 --gamma 0.50 --beta 100.0 --s 30 &&
# python3 comparison.py zlan ackley --d 100 --gamma 0.1 --beta 50.0 --s 5  --dir-type gaussian --h 0.001 &&
# python3 comparison.py cbo ackley --d 100 --popsize 25 --lam 10.0 --sigma 20.0 --dt 0.001 --alpha 100.0 &&
# python3 comparison.py pso ackley --d 100 --popsize 50 --inertia 0.775 &&
# python3 comparison.py de ackley --d 100 --popsize 25 --F 0.6 --CR 0.6 &&
# python3 comparison.py rs ackley --d 100 --popsize 20 --sigma 1.0 &&
# python3 comparison.py xnes ackley --d 100 --sigma 10.0 --popsize 10 --eta_mu 0.01 



python3 comparison.py fd ackley --d 100 --gamma 0.50 --beta 100.0 --s 30 &&
python3 comparison.py zlan ackley --d 100 --gamma 0.1 --beta 50.0 --s 5  --dir-type gaussian --h 0.001 &&
python3 comparison.py cbo ackley --d 100 --popsize 25 --lam 10.0 --sigma 20.0 --dt 0.001 --alpha 100.0 &&
python3 comparison.py pso ackley --d 100 --popsize 50 --inertia 0.775 &&
python3 comparison.py de ackley --d 100 --popsize 25 --F 0.6 --CR 0.6 &&
python3 comparison.py rs ackley --d 100 --popsize 20 --sigma 1.0 &&
python3 comparison.py xnes ackley --d 100 --sigma 10.0 --popsize 10 --eta_mu 0.01 