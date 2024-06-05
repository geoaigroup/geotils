"""
this file has all the libraries from geotorch imported this allows you to import them directly from geotils
if a new dataset is added to geotorch you should import it here and then it will be added to geotils
"""

from torchgeo.datasets import Sentinel1
from torchgeo.datasets import Sentinel2
from torchgeo.datasets import Landsat1
from torchgeo.datasets import Landsat2
from torchgeo.datasets import Landsat3
from torchgeo.datasets import SpaceNet1
from torchgeo.datasets import SpaceNet2
from torchgeo.datasets import SpaceNet3
from torchgeo.datasets import SpaceNet4
from torchgeo.datasets import SpaceNet5
from torchgeo.datasets import SpaceNet6
from torchgeo.datasets import SpaceNet7
from torchgeo.datasets import NAIP
from torchgeo.datasets import biomassters
from torchgeo.datasets import advance
from torchgeo.datasets import agb_live_woody_density
from torchgeo.datasets import benin_cashews
from torchgeo.datasets import (
    # cdf,
    cdl,
    chesapeake,
    cloud_cover,
    cms_mangrove_canopy,
    cyclone,
    cowc,
    # cv4a_kenya_crop_typec,
    cbf,
    dfc2022,
    deepglobelandcover,
    eddmaps,
    enviroatlas,
    esri2020,
    etci2021,
    eudem,
    eurosat,
    fair1m,
    fire_risk,
    forestdamage,
    gbif,
    geo,
    gid15,
    globbiomass,
    idtrees,
    inaturalist,
    inria,
    l7irish,
    l8biome,
    landcoverai,
    landsat,
    levircd,
    loveda,
    mapinwild,
    millionaid,
    nasa_marine_debris,
    nlcd,
    openbuildings,
    oscd,
    pastis,
    patternnet,
    potsdam,
    reforestree,
    resisc45,
    rwanda_field_boundary,
    seasonet,
    seco,
    sen12ms,
    sentinel,
    skippd,
    so2sat,
    spacenet,
    splits,
    ssl4eo,
    ssl4eo_benchmark,
    sustainbench_crop_yield,
    ucmerced,
    usavars,
    utils,
    western_usa_live_fuel_moisture,
    vaihingen,
    vhr10,
    WesternUSALiveFuelMoisture,
    agb_live_woody_density,
    xview,
    zuericrop,
)
