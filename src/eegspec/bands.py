from dataclasses import dataclass

# Default bands
DEFAULT_BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 7.0),
                 "lower_alpha": (8.0, 10.0),
                 "upper_alpha": (10.0, 12.0),
                 "beta": (13.0, 30.0),
                 "gamma": (30.0, 45.0)}
