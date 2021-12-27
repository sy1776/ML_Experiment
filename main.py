import time
from util import load_adult_data
from analyze_data import describe_data
from manipulate_data import *
from models import run_models

DISPLAY = True

def main():
    start_time = time.time()
    if DISPLAY:
        print("Start = %s" % (time.ctime()) )
        print("______________________________________________")
        print("")

    dataFrame = load_adult_data()
    describe_data(dataFrame)
    removedspaceDF = remove_space(dataFrame)
    filteredDF = filter_missing_data(removedspaceDF)
    nodupsDF = find_remove_dups(filteredDF)
    #transformedDF = transform_data(nodupsDF)
    transformedDF = encode_categorical_data(nodupsDF)
    run_models(transformedDF)
    duration = time.time() - start_time
    if DISPLAY:
        print("______________________________________________")
        print("")
        print("End= %s, Duration= %d seconds" % (time.ctime(), duration))

if __name__ == "__main__":
    main()