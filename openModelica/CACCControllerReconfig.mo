model CACCControllerReconfig
  // Parametri configurabili
  parameter Real c1 = 0.5;
  parameter Real eps = 1;
  parameter Real omega_n = 0.2;
  parameter Real targetDistance = 15;
  parameter Real initialAccel = 0;
  parameter Real simStep = 1;

  Real myTime;
  Real desiredAccelFallback;
  Real next_speed;

  // Ingressi
  Modelica.Blocks.Interfaces.RealInput ego_x;
  Modelica.Blocks.Interfaces.RealInput lead_x;
  Modelica.Blocks.Interfaces.RealInput ego_speed;
  Modelica.Blocks.Interfaces.RealInput lead_speed;
  Modelica.Blocks.Interfaces.RealInput accel_in;
  Modelica.Blocks.Interfaces.RealInput control_mode; // -1 = normale; > 0 = tempo in secondi per fermarsi

  // Uscita
  Modelica.Blocks.Interfaces.RealOutput desiredAcceleration;

initial equation
  myTime = 0.0;
equation 
  der(myTime) = simStep;

  // Modalità fallback (control_mode > 0): calcolo accelerazione per fermarsi
  desiredAccelFallback = if control_mode > 0 then -ego_speed / control_mode else 0;

  // Calcolo velocità prevista per evitare velocità negativa
  next_speed = ego_speed + desiredAccelFallback * 0.1;

  desiredAcceleration = 
    if time == 0 then
      initialAccel
    else if control_mode == -1 then
      // Modalità normale CACC
      2 * (1 - c1) * accel_in
      + (- (2 * eps - c1 * (eps + sqrt(eps * eps - 1))) * omega_n) * (ego_speed - lead_speed)
      + (- (c1 * (eps + sqrt(eps * eps - 1)) * omega_n)) * (ego_speed - lead_speed)
      + (- omega_n^2) * (ego_x - lead_x + targetDistance)
    else if next_speed <= 0 then
      // Velocità futura negativa: fermo
      0
   else
      // Modalità fallback: calcolo accelerazione per fermarsi entro `control_mode`
      desiredAccelFallback;

end CACCControllerReconfig;
