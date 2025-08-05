one_shot_example = """INPUT:charge conditions for fuel_consumption and aircraft_type:

for passenger the following charges apply:
  - if fuel_consumption in [500.00, 3710.42), charge is 0.0 × fuel_consumption
  - if fuel_consumption in [3710.42, 4284.00), charge is 1.0 × fuel_consumption
  - if fuel_consumption in [4284.00, 4487.57), charge is 2.0 × fuel_consumption
  - if fuel_consumption in [4487.57, 5000.00), charge is 3.0 × fuel_consumption
for cargo the following charges apply:
  - if fuel_consumption in [500.00, 3710.42), charge is 0.0 × fuel_consumption
  - if fuel_consumption in [3710.42, 4284.00), charge is 1.1 × fuel_consumption
  - if fuel_consumption in [4284.00, 4487.57), charge is 2.2 × fuel_consumption
  - if fuel_consumption in [4487.57, 5000.00), charge is 3.3 × fuel_consumption
for private the following charges apply:
  - if fuel_consumption in [500.00, 3710.42), charge is 0.0 × fuel_consumption
  - if fuel_consumption in [3710.42, 4284.00), charge is 1.2 × fuel_consumption
  - if fuel_consumption in [4284.00, 4487.57), charge is 2.4 × fuel_consumption
  - if fuel_consumption in [4487.57, 5000.00), charge is 3.6 × fuel_consumption
for military the following charges apply:
  - if fuel_consumption in [500.00, 3710.42), charge is 0.0 × fuel_consumption
  - if fuel_consumption in [3710.42, 4284.00), charge is 1.3 × fuel_consumption
  - if fuel_consumption in [4284.00, 4487.57), charge is 2.6 × fuel_consumption
  - if fuel_consumption in [4487.57, 5000.00), charge is 3.9 × fuel_consumption.
  
  OUTPUT:<table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; text-align: center; font-family: Arial, sans-serif; width: 100%;">
  <caption style="font-weight: bold; font-size: 1.2em; margin-bottom: 10px;">Fuel Consumption Charges by Aircraft Type</caption>
  <thead>
    <tr>
      <th><strong>Fuel Consumption Range (Gallons)</strong></th>
      <th><strong>Passenger Rate</strong></th>
      <th><strong>Cargo Rate</strong></th>
      <th><strong>Private Rate</strong></th>
      <th><strong>Military Rate</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>500.00 – 3710.42</td>
      <td>0.0 × fuel_consumption</td>
      <td>0.0 × fuel_consumption</td>
      <td>0.0 × fuel_consumption</td>
      <td>0.0 × fuel_consumption</td>
    </tr>
    <tr>
      <td>3710.42 – 4284.00</td>
      <td>1.0 × fuel_consumption</td>
      <td>1.1 × fuel_consumption</td>
      <td>1.2 × fuel_consumption</td>
      <td>1.3 × fuel_consumption</td>
    </tr>
    <tr>
      <td>4284.00 – 4487.57</td>
      <td>2.0 × fuel_consumption</td>
      <td>2.2 × fuel_consumption</td>
      <td>2.4 × fuel_consumption</td>
      <td>2.6 × fuel_consumption</td>
    </tr>
    <tr>
      <td>4487.57 – 5000.00</td>
      <td>3.0 × fuel_consumption</td>
      <td>3.3 × fuel_consumption</td>
      <td>3.6 × fuel_consumption</td>
      <td>3.9 × fuel_consumption</td>
    </tr>
  </tbody>
</table>
""" 