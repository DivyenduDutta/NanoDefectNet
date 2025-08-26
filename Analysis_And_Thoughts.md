#### Class percentage exploration

Class percentages for overall dataset:
{
    'Center_0': 77.62068965517241,
    'Donut_1': 0.0,
    'Edge-Loc_2': 12.241379310344827,
    'Edge-Ring_3': 0.8620689655172413,
    'Loc_4': 5.931034482758621,
    'Random_5': 1.8275862068965518,
    'Scratch_6': 0.7931034482758621,
    'Near-full_7': 0.7241379310344828,
    'none_8': 0.0
}


Class percentages for train dataset:
{
    'Center_0': 77.70935960591133,
    'Donut_1': 0.0,
    'Edge-Loc_2': 12.31527093596059,
    'Edge-Ring_3': 0.7389162561576355,
    'Loc_4': 6.280788177339902,
    'Random_5': 1.5394088669950738,
    'Scratch_6': 0.6773399014778325,
    'Near-full_7': 0.7389162561576355,
    'none_8': 0.0
}


Class percentages for validation dataset:
{
    'Center_0': 77.87356321839081,
    'Donut_1': 0.0,
    'Edge-Loc_2': 12.068965517241379,
    'Edge-Ring_3': 1.0057471264367817,
    'Loc_4': 4.885057471264368,
    'Random_5': 2.586206896551724,
    'Scratch_6': 0.8620689655172413,
    'Near-full_7': 0.7183908045977011,
    'none_8': 0.0
}

Class percentages for test dataset:
{
    'Center_0': 77.06896551724138,
    'Donut_1': 0.0,
    'Edge-Loc_2': 12.241379310344827,
    'Edge-Ring_3': 1.0344827586206897,
    'Loc_4': 6.206896551724138,
    'Random_5': 1.7241379310344827,
    'Scratch_6': 1.0344827586206897,
    'Near-full_7': 0.6896551724137931,
    'none_8': 0.0
}

Even though we did not explicitly try to ensure that the class percentages sync up when splitting into train, val, test datasets, it still matches up with the overall dataset. I think this is a result of the dataset being large enough.
