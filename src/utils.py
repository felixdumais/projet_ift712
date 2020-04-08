def unique(list1: list):
    """
    Function that return a list with unique elements

    :arg
        list1 (list): a python list

    :return
        unique_list (list): List containing unique element
    """
    # intilize a null list
    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
            # print list
    return unique_list
