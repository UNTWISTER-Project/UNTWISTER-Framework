model LeadCarV3WithNoise
  EgoWithNoiseWithStopSignal egoWithNoise(v0 = 8.33, x0 = 30)  annotation(
    Placement(transformation(origin = {-18, 0}, extent = {{-24, -24}, {24, 24}})));
  // input
  Modelica.Blocks.Interfaces.RealInput inputAccel annotation(
    Placement(transformation(origin = {-110, 2}, extent = {{-20, -20}, {20, 20}}), iconTransformation(origin = {-90, 10}, extent = {{-20, -20}, {20, 20}})));
  Modelica.Blocks.Interfaces.RealInput inputMode annotation(
    Placement(transformation(origin = {-112, -32}, extent = {{-20, -20}, {20, 20}}), iconTransformation(origin = {-100, 10}, extent = {{-20, -20}, {20, 20}})));
  // output
  Modelica.Blocks.Interfaces.RealOutput x annotation(
    Placement(transformation(origin = {120, 86}, extent = {{-20, -20}, {20, 20}}), iconTransformation(origin = {108, 66}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Interfaces.RealOutput speed annotation(
    Placement(transformation(origin = {120, 24}, extent = {{-20, -20}, {20, 20}}), iconTransformation(origin = {106, -74}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Interfaces.RealOutput outputAccel annotation(
    Placement(transformation(origin = {123, -43}, extent = {{-23, -23}, {23, 23}}), iconTransformation(origin = {108, -90}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Interfaces.RealOutput noisyX annotation(
    Placement(transformation(origin = {120, 56}, extent = {{-20, -20}, {20, 20}}), iconTransformation(origin = {108, 66}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Interfaces.RealOutput noisySpeed annotation(
    Placement(transformation(origin = {120, -8}, extent = {{-20, -20}, {20, 20}}), iconTransformation(origin = {106, -74}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Blocks.Interfaces.RealOutput noisyOutputAccel annotation(
    Placement(transformation(origin = {122, -81}, extent = {{-23, -23}, {23, 23}}), iconTransformation(origin = {108, -90}, extent = {{-10, -10}, {10, 10}})));
    // === Debug ===
  //Modelica.Blocks.Interfaces.RealOutput countdownTimer_out;
equation
  connect(inputAccel, egoWithNoise.accel) annotation(
    Line(points = {{-110, 2}, {-40, 2}}, color = {0, 0, 127}));
  connect(egoWithNoise.noisyX, noisyX) annotation(
    Line(points = {{8, 14}, {120, 14}, {120, 56}}, color = {0, 0, 127}));
  connect(egoWithNoise.noisyAccel_out, noisyOutputAccel) annotation(
    Line(points = {{8, 0}, {122, 0}, {122, -80}}, color = {0, 0, 127}));
  connect(egoWithNoise.x, x) annotation(
    Line(points = {{8, 14}, {120, 14}, {120, 86}}, color = {0, 0, 127}));
  connect(egoWithNoise.noisySpeed, noisySpeed) annotation(
    Line(points = {{8, 2}, {120, 2}, {120, -8}}, color = {0, 0, 127}));
  connect(egoWithNoise.speed, speed) annotation(
    Line(points = {{8, 2}, {120, 2}, {120, 24}}, color = {0, 0, 127}));
  connect(egoWithNoise.accel_out, outputAccel) annotation(
    Line(points = {{8, 0}, {124, 0}, {124, -42}}, color = {0, 0, 127}));
  connect(inputMode, egoWithNoise.deceleration) annotation(
    Line(points = {{-112, -32}, {-40, -32}, {-40, -10}}, color = {0, 0, 127}));
  //connect(countdownTimer_out, egoWithNoise.countdownTimer_out);
  annotation(
    uses(Modelica(version = "4.0.0")),
    Diagram);
end LeadCarV3WithNoise;