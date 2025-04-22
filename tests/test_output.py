import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))



from mymodels.output import Output



def test_output():
    



    return None



if __name__ == "__main__":
    output = Output(
        results_dir = "./results/test_output"
    )

    test_output()
