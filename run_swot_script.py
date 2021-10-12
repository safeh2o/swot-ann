import sys
from swotann.nnetwork import NNetwork


def run_swot():
    SWOT_net = NNetwork()
    input_file = sys.argv[1]
    results_file = sys.argv[3]
    report_file = sys.argv[4]
    storage_target = sys.argv[5]
    SWOT_net.run_swot(input_file, results_file, report_file, storage_target)


if __name__ == "__main__":
    run_swot()
