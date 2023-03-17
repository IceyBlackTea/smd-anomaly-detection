def read_data_file(file_path):
    with open(file_path) as file:
        lines = file.readlines()

    table = []
    for line in lines:
        table.append([])
        vars = line.split(",")
        for var in vars:
            table[-1].append(float(var))

    return table


def read_label_file(file_path):
    with open(file_path) as file:
        lines = file.readlines()

    table = []
    for line in lines:
        table.append(int(line))

    return table


def split_table(table):
    split_dimensions = []
    for i in range(len(table[0])):
        split_dimensions.append([])
        for line in table:
            split_dimensions[-1].append(line[i])

    return split_dimensions


def split_vec_table(table):
    split_dimensions = []
    for i in range(len(table[0])):
        split_dimensions.append([])
        for line in table:
            split_dimensions[-1].append([line[i]])

    return split_dimensions


def read_interpretation_file(file_path):
    with open(file_path) as file:
        lines = file.readlines()
    
    table = {}
    dims = []

    for line in lines:
        line_splits = line[:-1].split(":")
        time_part = line_splits[0]
        dim_part = line_splits[1]

        start_time = int(time_part.split("-")[0])
        end_time = int(time_part.split("-")[1])
        
        if dim_part not in dims:
            dims.append(dim_part)

        dim_strs = dim_part.split(",")
        for dim_str in dim_strs:
            if int(dim_str)-1 not in table:
                table[int(dim_str)-1] = []

            table[int(dim_str)-1].extend(range(start_time, end_time))

    return table, dims
