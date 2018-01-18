import sys
from process_data import InformatiCup2018

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print("Supply at least two parameters, data dir and route csv file. Third optionally for custom export csv file.")
        exit(0)

    data_dir = sys.argv[1]
    route_file = sys.argv[2]

    if len(sys.argv) == 4:
        route_output = sys.argv[3]

    ic = InformatiCup2018(data_dir=data_dir)
    ic.calculate_tank_stops(route_file)
