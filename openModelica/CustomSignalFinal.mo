model CustomSignalFromTimestamp
  // Durate degli intervalli nulli
  parameter Real T_null[14] = {2.0, 4.9, 8.2, 6.0, 5.9, 7.1, 4.2, 7.6, 6.9, 8.8, 4.7, 4.4, 6.7, 2.3};

  // Semiperiodi positivi
  parameter Real A_pos[7] = {1.4, 2.28, 1.47, 1.88, 2.16, 2.17, 1.83};
  parameter Real f_pos[7] = {0.32, 0.33, 0.32, 0.39, 0.37, 0.28, 0.38};

  // Semiperiodi negativi
  parameter Real A_neg[6] = {1.6, 1.57, 1.24, 2.12, 1.33, 1.90};
  parameter Real f_neg[6] = {0.34, 0.36, 0.33, 0.23, 0.33, 0.27};

  // Timestamp da cui parte la simulazione
  parameter Real currentTimestamp = 0;

  // Calcolo tempi assoluti sequenza originale
  Real t0 = 0;
  Real t1 = t0 + T_null[1];
  Real t2 = t1 + 1/(2*f_pos[1]);
  Real t3 = t2 + T_null[2];
  Real t4 = t3 + 1/(2*f_neg[1]);
  Real t5 = t4 + T_null[3];
  Real t6 = t5 + 1/(2*f_pos[2]);
  Real t7 = t6 + T_null[4];
  Real t8 = t7 + 1/(2*f_neg[2]);
  Real t9 = t8 + T_null[5];
  Real t10 = t9 + 1/(2*f_pos[3]);
  Real t11 = t10 + T_null[6];
  Real t12 = t11 + 1/(2*f_neg[3]);
  Real t13 = t12 + T_null[7];
  Real t14 = t13 + 1/(2*f_pos[4]);
  Real t15 = t14 + T_null[8];
  Real t16 = t15 + 1/(2*f_neg[4]);
  Real t17 = t16 + T_null[9];
  Real t18 = t17 + 1/(2*f_pos[5]);
  Real t19 = t18 + T_null[10];
  Real t20 = t19 + 1/(2*f_neg[5]);
  Real t21 = t20 + T_null[11];
  Real t22 = t21 + 1/(2*f_pos[6]);
  Real t23 = t22 + T_null[12];
  Real t24 = t23 + 1/(2*f_neg[6]);
  Real t25 = t24 + T_null[13];
  Real t26 = t25 + 1/(2*f_pos[7]);
  Real t27 = t26 + T_null[14];

  // Flags booleani per verificare se currentTimestamp è dentro un semiperiodo
  Boolean insidePos1 = currentTimestamp >= t1 and currentTimestamp < t2;
  Boolean insideNeg1 = currentTimestamp >= t3 and currentTimestamp < t4;
  Boolean insidePos2 = currentTimestamp >= t5 and currentTimestamp < t6;
  Boolean insideNeg2 = currentTimestamp >= t7 and currentTimestamp < t8;
  Boolean insidePos3 = currentTimestamp >= t9 and currentTimestamp < t10;
  Boolean insideNeg3 = currentTimestamp >= t11 and currentTimestamp < t12;
  Boolean insidePos4 = currentTimestamp >= t13 and currentTimestamp < t14;
  Boolean insideNeg4 = currentTimestamp >= t15 and currentTimestamp < t16;
  Boolean insidePos5 = currentTimestamp >= t17 and currentTimestamp < t18;
  Boolean insideNeg5 = currentTimestamp >= t19 and currentTimestamp < t20;
  Boolean insidePos6 = currentTimestamp >= t21 and currentTimestamp < t22;
  Boolean insideNeg6 = currentTimestamp >= t23 and currentTimestamp < t24;
  Boolean insidePos7 = currentTimestamp >= t25 and currentTimestamp < t26;

  // Offset per la fase (tempo già trascorso nel semiperiodo)
  Real offsetPos1 = if insidePos1 then currentTimestamp - t1 else 0;
  Real offsetNeg1 = if insideNeg1 then currentTimestamp - t3 else 0;
  Real offsetPos2 = if insidePos2 then currentTimestamp - t5 else 0;
  Real offsetNeg2 = if insideNeg2 then currentTimestamp - t7 else 0;
  Real offsetPos3 = if insidePos3 then currentTimestamp - t9 else 0;
  Real offsetNeg3 = if insideNeg3 then currentTimestamp - t11 else 0;
  Real offsetPos4 = if insidePos4 then currentTimestamp - t13 else 0;
  Real offsetNeg4 = if insideNeg4 then currentTimestamp - t15 else 0;
  Real offsetPos5 = if insidePos5 then currentTimestamp - t17 else 0;
  Real offsetNeg5 = if insideNeg5 then currentTimestamp - t19 else 0;
  Real offsetPos6 = if insidePos6 then currentTimestamp - t21 else 0;
  Real offsetNeg6 = if insideNeg6 then currentTimestamp - t23 else 0;
  Real offsetPos7 = if insidePos7 then currentTimestamp - t25 else 0;
  
  // Tempi relativi rispetto a currentTimestamp, con gestione inside (inizio relativo = 0 se dentro)
  // 1° semiperiodo positivo
  Real t1_rel = if insidePos1 then 0 else t1 - currentTimestamp;
  Real t2_rel = t1_rel + 1/(2*f_pos[1]) - offsetPos1;

  // 1° semiperiodo negativo
  Real t3_rel = if insidePos1 then t2_rel + T_null[2]
                         else if insideNeg1 then 0
                         else t3 - currentTimestamp;
  Real t4_rel = t3_rel + 1/(2*f_neg[1]) - offsetNeg1;

  // 2° semiperiodo positivo
  Real t5_rel = if insideNeg1 then t4_rel + T_null[3]
                         else if insidePos2 then 0
                         else t5 - currentTimestamp;
  Real t6_rel = t5_rel + 1/(2*f_pos[2]) - offsetPos2;

  // 2° semiperiodo negativo
  Real t7_rel = if insidePos2 then t6_rel + T_null[4]
                         else if insideNeg2 then 0
                         else t7 - currentTimestamp;
  Real t8_rel = t7_rel + 1/(2*f_neg[2]) - offsetNeg2;

  // 3° semiperiodo positivo
  Real t9_rel = if insideNeg2 then t8_rel + T_null[5]
                         else if insidePos3 then 0
                         else t9 - currentTimestamp;
  Real t10_rel = t9_rel + 1/(2*f_pos[3]) - offsetPos3;

  // 3° semiperiodo negativo
  Real t11_rel = if insidePos3 then t10_rel + T_null[6]
                          else if insideNeg3 then 0
                          else t11 - currentTimestamp;
  Real t12_rel = t11_rel + 1/(2*f_neg[3]) - offsetNeg3;

  // 4° semiperiodo positivo
  Real t13_rel = if insideNeg3 then t12_rel + T_null[7]
                          else if insidePos4 then 0
                          else t13 - currentTimestamp;
  Real t14_rel = t13_rel + 1/(2*f_pos[4]) - offsetPos4;

  // 4° semiperiodo negativo
  Real t15_rel = if insidePos4 then t14_rel + T_null[8]
                          else if insideNeg4 then 0
                          else t15 - currentTimestamp;
  Real t16_rel = t15_rel + 1/(2*f_neg[4]) - offsetNeg4;

  // 5° semiperiodo positivo
  Real t17_rel = if insideNeg4 then t16_rel + T_null[9]
                          else if insidePos5 then 0
                          else t17 - currentTimestamp;
  Real t18_rel = t17_rel + 1/(2*f_pos[5]) - offsetPos5;

  // 5° semiperiodo negativo
  Real t19_rel = if insidePos5 then t18_rel + T_null[10]
                          else if insideNeg5 then 0
                          else t19 - currentTimestamp;
  Real t20_rel = t19_rel + 1/(2*f_neg[5]) - offsetNeg5;

  // 6° semiperiodo positivo
  Real t21_rel = if insideNeg5 then t20_rel + T_null[11]
                          else if insidePos6 then 0
                          else t21 - currentTimestamp;
  Real t22_rel = t21_rel + 1/(2*f_pos[6]) - offsetPos6;

  // 6° semiperiodo negativo
  Real t23_rel = if insidePos6 then t22_rel + T_null[12]
                          else if insideNeg6 then 0
                          else t23 - currentTimestamp;
  Real t24_rel = t23_rel + 1/(2*f_neg[6]) - offsetNeg6;

  // 7° semiperiodo positivo
  Real t25_rel = if insideNeg6 then t24_rel + T_null[13]
                          else if insidePos7 then 0
                          else t25 - currentTimestamp;
  Real t26_rel = t25_rel + 1/(2*f_pos[7]) - offsetPos7;

  Modelica.Blocks.Interfaces.RealOutput acc annotation(
    Placement(visible=true, transformation(origin={110,12}, extent={{-10,-10},{10,10}}, rotation=0)));

equation
  acc = if time < t1_rel then 0
        else if time < t2_rel then A_pos[1]*sin(2*Modelica.Constants.pi*f_pos[1]*(time - t1_rel + offsetPos1))
        else if time < t3_rel then 0
        else if time < t4_rel then -A_neg[1]*sin(2*Modelica.Constants.pi*f_neg[1]*(time - t3_rel + offsetNeg1))
        else if time < t5_rel then 0
        else if time < t6_rel then A_pos[2]*sin(2*Modelica.Constants.pi*f_pos[2]*(time - t5_rel + offsetPos2))
        else if time < t7_rel then 0
        else if time < t8_rel then -A_neg[2]*sin(2*Modelica.Constants.pi*f_neg[2]*(time - t7_rel + offsetNeg2))
        else if time < t9_rel then 0
        else if time < t10_rel then A_pos[3]*sin(2*Modelica.Constants.pi*f_pos[3]*(time - t9_rel + offsetPos3))
        else if time < t11_rel then 0
        else if time < t12_rel then -A_neg[3]*sin(2*Modelica.Constants.pi*f_neg[3]*(time - t11_rel + offsetNeg3))
        else if time < t13_rel then 0
        else if time < t14_rel then A_pos[4]*sin(2*Modelica.Constants.pi*f_pos[4]*(time - t13_rel + offsetPos4))
        else if time < t15_rel then 0
        else if time < t16_rel then -A_neg[4]*sin(2*Modelica.Constants.pi*f_neg[4]*(time - t15_rel + offsetNeg4))
        else if time < t17_rel then 0
        else if time < t18_rel then A_pos[5]*sin(2*Modelica.Constants.pi*f_pos[5]*(time - t17_rel + offsetPos5))
        else if time < t19_rel then 0
        else if time < t20_rel then -A_neg[5]*sin(2*Modelica.Constants.pi*f_neg[5]*(time - t19_rel + offsetNeg5))
        else if time < t21_rel then 0
        else if time < t22_rel then A_pos[6]*sin(2*Modelica.Constants.pi*f_pos[6]*(time - t21_rel + offsetPos6))
        else if time < t23_rel then 0
        else if time < t24_rel then -A_neg[6]*sin(2*Modelica.Constants.pi*f_neg[6]*(time - t23_rel + offsetNeg6))
        else if time < t25_rel then 0
        else if time < t26_rel then A_pos[7]*sin(2*Modelica.Constants.pi*f_pos[7]*(time - t25_rel + offsetPos7))
        else 0;
end CustomSignalFromTimestamp;
