"""Scripts de analisis post-corrida (regimen MaxFES).

- ``normality``: Shapiro-Wilk + Anderson-Darling sobre el fitness final por funcion
  (lee ``summary.csv``).
- ``decoder_collapse``: auditoria del decoder TMLAP (uniqueness + perturb sensitivity).

Los scripts ligados al esquema antiguo (eventos por iteracion / stagnation_length /
post_status) se eliminaron al migrar el controlador a por-agente + FES. Las
estadisticas agregadas (mean/median/std del gap por ``problem x max_fes``) ahora
las producen directamente los runners en ``values/statistics.csv``.
"""
