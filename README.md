# Siyu_DWR_Fingerprinting
 Projects in collaboration with the California Department of Water Resources. Apply machine learning for X2 estimation and fingerprinting.
 ## What's in this repo
 * [FP_Stations.pdf](Siyu_DWR_Fingerprinting/FP_Stations.pdf): the map showing 19 locations of interest.
 * [Fingerprinting_Readings](Siyu_DWR_Fingerprinting/Fingerprinting_Readings): Reading materials.
   - [Ref2](Siyu_DWR_Fingerprinting/Fingerprinting_Readings/Ref2_2018_DSM2_Emulation_Chen et al.pdf): the paper that proposes training one ANN per location per boundary source. The used input variables are (according to Table 4 in the paper):
   <!-- <style type="text/css">
   .tg  {border-collapse:collapse;border-spacing:0;}
   .tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
     overflow:hidden;padding:10px 5px;word-break:normal;}
   .tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
     font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
   .tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
   </style> -->
   <table class="tg">
   <thead>
     <tr>
       <th class="tg-0pky">Number</th>
       <th class="tg-0pky">Input variable</th>
       <th class="tg-0pky">Units</th>
     </tr>
   </thead>
   <tbody>
     <tr>
       <td colspan="3" class="tg-0pky"><em>Flow &amp; water level</em></td>
     </tr>
     <tr>
       <td class="tg-0pky">1</td>
       <td class="tg-0pky">Sacramento River flow at Freeport</td>
       <td class="tg-0pky">cfs</td>
     </tr>
     <tr>
       <td class="tg-0pky">2</td>
       <td class="tg-0pky">San Joaquin River flow at Vernalis</td>
       <td class="tg-0pky">cfs</td>
     </tr>
     <tr>
       <td class="tg-0pky">3</td>
       <td class="tg-0pky">Mokelumne River flow</td>
       <td class="tg-0pky">cfs</td>
     </tr>
     <tr>
       <td class="tg-0pky">4</td>
       <td class="tg-0pky">Cosumnes River flow</td>
       <td class="tg-0pky">cfs</td>
     </tr>
     <tr>
       <td class="tg-0pky">5</td>
       <td class="tg-0pky">Calaveras River flow</td>
       <td class="tg-0pky">cfs</td>
     </tr>
     <tr>
       <td class="tg-0pky">6</td>
       <td class="tg-0pky">Yolo Bypass flow</td>
       <td class="tg-0pky">cfs</td>
     </tr>
     <tr>
       <td class="tg-0pky">7</td>
       <td class="tg-0pky">Delta island net channel depletions</td>
       <td class="tg-0pky">cfs</td>
     </tr>
     <tr>
       <td class="tg-0pky">8</td>
       <td class="tg-0pky">Delta combined exports</td>
       <td class="tg-0pky">cfs</td>
     </tr>
     <tr>
       <td class="tg-0pky">9</td>
       <td class="tg-0pky">Martinez tide</td>
       <td class="tg-0pky">ft MSL</td>
     </tr>
     <tr>
       <td class="tg-0pky" colspan="3"><span style="font-style:italic"><em>Gates &amp; barrier operations</em></span></td>
     </tr>
     <tr>
       <td class="tg-0pky">10</td>
       <td class="tg-0pky">Grant Line Canal</td>
       <td class="tg-0pky">0, 1</td>
     </tr>
     <tr>
       <td class="tg-0pky">11</td>
       <td class="tg-0pky">Middle River near Tracy</td>
       <td class="tg-0pky">0, 1</td>
     </tr>
     <tr>
       <td class="tg-0pky">12</td>
       <td class="tg-0pky">Old River at Tracy</td>
       <td class="tg-0pky">0, 1</td>
     </tr>
     <tr>
       <td class="tg-0pky">13</td>
       <td class="tg-0pky">Head of Old River</td>
       <td class="tg-0pky">0, 1</td>
     </tr>
     <tr>
       <td class="tg-0pky">14</td>
       <td class="tg-0pky">Delta Cross Channel (DCC)</td>
       <td class="tg-0pky">0, 1, 2*</td>
     </tr>
   </tbody>
   </table>
   * 0: Gate is fully closed; 1: one gate is fully open; 2: two gates are fully open.

 * [Data](Siyu_DWR_Fingerprinting/Data): all the output data. Note that the column names in each csv file represent one location of interest in the Delta, as marked in the [map](Siyu_DWR_Fingerprinting/FP_Stations.pdf).
    - Six csv files: each for one boundary source.
    - [All.csv](Siyu_DWR_Fingerprinting/Data/All.csv): summation of the six boundary sources.
    - [dsm2_ann_inputs_historical_ec.xlsx](Siyu_DWR_Fingerprinting/Data/dsm2_ann_inputs_historical_ec.xlsx): Input dataset, including input variables:
      - #2, #8, #9, #14
      - Northern flow = #1 + #3 + #4 + #5 + #6
      - Delta consumptive usage (input #7?)
      - San Joaquin River inflow salinity at Vernalis.

 * [Train_Fingerprinting_ANN.ipynb](Siyu_DWR_Fingerprinting/Train_Fingerprinting_ANN.ipynb): Colab training script, same architectures as Delta modelling ANNs. Outputs are flattened into an $114 \times 1$ vector.
   - Usage:
        1. In your Google Drive, create a folder named "fingerprinting".
        1. Upload the folder "[Data](Siyu_DWR_Fingerprinting/Data)", this script, along with the helper script "[preprocessing_utils.py](Siyu_DWR_Fingerprinting/preprocessing_utils.py)". The directory structure shall look like:
   ```bash
fingerprinting
   ├── Data
   │   ├── dsm2_ann_inputs_historical_ec.xlsx
   │   ├── Ag.csv
   │   ├── All.csv
   │   ├── East.csv
   │   ├── Jones.csv
   │   ├── MTZ.csv
   │   ├── SAC.csv
   │   └── SJR.csv
   ├── preprocessing_utils.py
   └── Train_Fingerprinting_ANN.ipynb
   ```
    3. Open [Train_Fingerprinting_ANN](Siyu_DWR_Fingerprinting/Train_Fingerprinting_ANN.ipynb) via Google Colab and run it from there.

## A few thoughts:
 * Output scaling: still linearly normalize each variable individually?
 * Output arrangement: still treat them as a flattened vector?
   - This project is about identifying the volume contributed by 6 sources at 19 locations, they should be internally correlated.
