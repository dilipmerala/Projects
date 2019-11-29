library(jsonlite)
library(geosphere)
library(tidyverse)
library(leaflet)
library(ggplot)
library(ggmap)
library(spatstat)
library(osrm)

#Setting up working directory
setwd("C:/Users/abhip/OneDrive/Desktop")
#Reading csv file
London_csv <- read.csv("London_UM.csv")
# Checking the structure of data
str(London_csv)
# Loading objects
my_london_regions <- jsonlite::fromJSON("london_GB.json")
# check your region list
head(my_london_regions$features$properties)
# polygon coordinates for each region
str(my_london_regions$features$geometry$coordinates)

#Calculating centroid of 1 region
my_london_polygons=my_london_regions$features$geometry$coordinates

#Creating an empty dataframe
dat <- data.frame()
#Looping to store centroids of all the observations
for (i in 1: length(my_london_polygons)){   
  my_temp_poly<-my_london_polygons[[i]]
  poly_len <- length(my_temp_poly)/2
  poly_df <- data.frame (lng = my_temp_poly[1,1,1:poly_len,1], lat = my_temp_poly[1,1,1:poly_len,2])
  my_poly_matrix <- data.matrix(poly_df)
  temp_centroid <- data.frame()
  temp_centroid <- centroid(my_poly_matrix)
  temp_centroid[[3]] <- i
  dat <- rbind(dat, temp_centroid)
}
colnames(dat) <- c("lng", "lat", "id")
dat <- dat[c("id", "lat", "lng")]
#Plotting regions on graph

leaflet(dat) %>% 
  addTiles() %>% 
  addMarkers() %>% 
  addPolygons(lng= poly_df$lng, lat=poly_df$lat)

#calculate distance
my_route_d<-osrmRoute(src=my_london_centroids[my_r2,] , dst=my_london_centroids[my_r3,], overview = FALSE)
# route segments if needed to draw a polyline
my_route<-osrmRoute(src=dat[1,] , dst=dat[23,],overview = FALSE)



