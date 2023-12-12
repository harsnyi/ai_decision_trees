import numpy as np #(működik a Moodle-ben is)


######################## 1. feladat, entrópiaszámítás #########################
def get_entropy(n_cat1: int, n_cat2: int) -> float:
    entropy = 0
    
    total = n_cat1 + n_cat2
    p_cat1 = n_cat1 / total
    p_cat2 = n_cat2 / total

    entropy = - p_cat1 * np.log2(p_cat1) - p_cat2 * np.log2(p_cat2) if p_cat1 != 0 and p_cat2 != 0 else 0
    return entropy
    

###################### 2. feladat, optimális szeparáció #######################
def get_best_separation(features: list, labels: list) -> (int, int):
    best_separation_feature, best_separation_value = 0, 0
    #TODO számítsa ki a legjobb szeparáció tulajdonságát és értékét!
    return best_separation_feature, best_separation_value

################### 3. feladat, döntési fa implementációja ####################
def main():
    #TODO: implementálja a döntési fa tanulását!
    return 0

if __name__ == "__main__":
    main()
