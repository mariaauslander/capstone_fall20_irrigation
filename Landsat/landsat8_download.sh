#!/bin/bash
# install package
pip install landsatxplore
landsatxplore --help

# credentials
export LANDSATXPLORE_USERNAME=<your_username>
export LANDSATXPLORE_PASSWORD=<your_password>

# search data
landsatxplore search --dataset LANDSAT_8_C1 --location 37.125 -121.125 --start 2020-01-01 --end 2020-10-31

# download data
landsatxplore download LC80430342020007LGN00
landsatxplore download LC80440342020014LGN00
landsatxplore download LC80430342020023LGN00
landsatxplore download LC80440342020030LGN00
landsatxplore download LC80430342020039LGN00
landsatxplore download LC80440342020046LGN00 
landsatxplore download LC80430342020055LGN00
landsatxplore download LC80440342020062LGN00
landsatxplore download LC80430342020071LGN00
landsatxplore download LC80440342020078LGN00
landsatxplore download LC80430342020087LGN00
landsatxplore download LC80440342020094LGN00
landsatxplore download LC80430342020103LGN00
landsatxplore download LC80440342020110LGN00
landsatxplore download LC80430342020119LGN00
landsatxplore download LC80440342020126LGN00
landsatxplore download LC80430342020135LGN00
landsatxplore download LC80440342020142LGN00
landsatxplore download LC80430342020151LGN00
landsatxplore download LC80440342020158LGN00
