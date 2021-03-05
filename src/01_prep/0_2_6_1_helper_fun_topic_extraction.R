# Denis' helper function to extract and interpret topics from BERT PCA/clusters
# "exploratory_topic_analysis_aux.R"

data_summary <- function(data, varname, groupnames){
  require(plyr)
  summary_func <- function(x, col){
    c(mean = mean(x[[col]], na.rm=TRUE),
      sd = sd(x[[col]], na.rm=TRUE))
  }
  data_sum<-ddply(data, groupnames, .fun=summary_func,
                  varname)
  data_sum <- rename(data_sum, c("mean" = varname))
  return(data_sum)
}



interpret = function(PCA.coordinates,selected.dimension,encodings,nn = 5){
  redundancies = which(duplicated(encodings$title_en))
  PCA.coordinates$index = 1:nrow(encodings)
  PCA.coordinates = PCA.coordinates[-redundancies,]
  PCA.coordinates = PCA.coordinates[order(PCA.coordinates[,selected.dimension]),]
  extremes = c(1:nn,(nrow(PCA.coordinates)-(nn-1)):nrow(PCA.coordinates))
  target = PCA.coordinates$index[extremes]
  output = data.frame(sign = c(rep("-",nn),rep("+",nn)),title = encodings$title_en[target],coordinate = PCA.coordinates[extremes,selected.dimension] )
  return(output)
}



find.closest = function(vec,dd2,z){
  dd3 = dd2[order(dd2[,vec]),"index"]
  dd4 = dd2[order(dd2[,vec]),vec]
  dd3 = dd3[1:z]
  dd4 = dd4[1:z]
  dd3 = cbind(dd3, dd4)
  return(dd3)
}



find.typical.titles = function(dims,centers,z,uw3){
  require(distances)
  dims2 = rbind(dims,centers)
  dd = distances(dims2)
  dd2 = as.data.frame(distance_columns(dd, c((nrow(dims)+1):(nrow(dims)+nrow(centers))),  row_indices = c(1:nrow(dims))))
  dd2$index = 1:nrow(dd2)
  
  closest = lapply(c(1:nrow(centers)),find.closest,dd2=dd2,z)
  closest = do.call(cbind,closest)
  
  closest.titles = as.data.frame(closest)
  names(closest.titles) <- rep(1:nrow(centers), each = 2)
  
  for (i in seq(1, ncol(closest), 2)){
    closest.titles[,i]=uw3$title_en[closest[,i]]
  }
  
  return(closest.titles)
}
