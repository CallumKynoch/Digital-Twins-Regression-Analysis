def calculate_equation_for_two_independent_variables():
    d = 3
    formula = "y = "
    for i1 in range(d + 1):
        for i2 in range(d - i1 + 1):
            formula += f"β_{i1}{i2}"
            formula += (f" x_1^{i1}" if i1 > 0 else "") + \
                       (f" x_2^{i2}" if i2 > 0 else "") + " + "

    formula += "ε"
    print(formula)


def calculate_equation_for_three_independent_variables():
    d = 3
    formula = "y = "
    for i1 in range(d + 1):
        for i2 in range(d - i1 + 1):
            for i3 in range(d - i1 - i2 + 1):
                formula += f"β_{i1}{i2}{i3}"
                formula += (f" x_1^{i1}" if i1 > 0 else "") + \
                           (f" x_2^{i2}" if i2 > 0 else "") + \
                           (f" x_3^{i3}" if i3 > 0 else "") + " + "

    formula += "ε"
    print(formula)


def calculate_equation_for_four_independent_variables():
    d = 3
    formula = "y = "
    for i1 in range(d + 1):
        for i2 in range(d - i1 + 1):
            for i3 in range(d - i1 - i2 + 1):
                for i4 in range(d - i1 - i2 - i3 + 1):
                    formula += f"β_{i1}{i2}{i3}{i4}"
                    formula += (f" x_1^{i1}" if i1 > 0 else "") + \
                               (f" x_2^{i2}" if i2 > 0 else "") + \
                               (f" x_3^{i3}" if i3 > 0 else "") + \
                               (f" x_4^{i4}" if i4 > 0 else "") + " + "

    formula += "ε"
    print(formula)


# Change function as required
calculate_equation_for_two_independent_variables()
