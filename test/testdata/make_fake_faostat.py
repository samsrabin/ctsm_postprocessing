"""
Make a CSV file in the style of FAOSTAT data, for testing
"""

# pylint: disable=invalid-name

import pandas as pd

# Define unique values for each column
uniq_areas = ["USA", "Germany", "China", "China, mainland", "China, Hong Kong SAR"]
uniq_years = [1986, 1987, 1988, 1989]
uniq_elements = ["Production", "Area harvested"]
uniq_crops = ["Maize", "Maize, green", "Rice", "Seed cotton, unginned"]

# Build the columns
areas = []
years = []
elements = []
crops = []
units = []
values = []
value = 0.5
for area in uniq_areas:
    for year in uniq_years:
        for element in uniq_elements:
            if element == "Production":
                unit = "t"
            elif element == "Area harvested":
                unit = "ha"
            else:
                raise RuntimeError(f"What unit for {element}?")
            for crop in uniq_crops:
                areas.append(area)
                years.append(year)
                elements.append(element)
                crops.append(crop)
                units.append(unit)
                value += 1
                values.append(value)

# Make the DataFrame
df = pd.DataFrame(
    {
        "Area": areas,
        "Year": years,
        "Element": elements,
        "Item": crops,
        "Unit": units,
        "Value": values,
    }
)
df.to_csv("test/testdata/test_faostat.csv", index=False)
