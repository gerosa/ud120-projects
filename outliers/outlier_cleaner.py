#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """

    
    cleaned_data = [(a[0], n[0], p[0] - n[0]) for p, a, n in zip(predictions, ages, net_worths)]
    cleaned_data = sorted(cleaned_data, key=lambda d: abs(d[2]))

    print("before cleaning: {}".format(len(cleaned_data)))

    cleaned_data = cleaned_data[:int(len(ages)*0.9)]
    print("after cleaning: {}".format(len(cleaned_data)))

    
    return cleaned_data

