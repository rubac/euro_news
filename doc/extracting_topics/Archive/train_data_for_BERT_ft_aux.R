tt.to.df = function(tt,keep=0,nom="domain"){
  if(keep==0){keep = 1:length(tt)}
  if(length(keep)==1){keep = 1:keep}
  tt = as.data.frame(tt)
  tt = tt[order(-tt$Freq)[keep],]
  colnames(tt)[1] = nom
  return(tt)
}

get_cases = function(e,tagz,data_tt,echantillon0,nn=nn){

  echantillon0[,"#1 ID"]=e
  echantillon0[,"#1 String"]=tagz[e]
  
  for.regex = tagz[e]
  for.regex = gsub("\\(","[(]",for.regex)
  for.regex = gsub("\\)","[)]",for.regex)
  
  where = str_detect(data_tt$Tags,paste0("(^|;( )*)",for.regex,"($|( )*;)"))
  
  if(length(which(where))==0|length(which(!where))==0){echantillon0=NULL}else{

    match = sample(which(where),nn,replace=TRUE)
    echantillon0[1:nn,c("#2 ID","#2 String")] = data_tt[match,c("index","Title")]
    echantillon0[1:nn,c("Quality")]=1
  
    unmatch = sample(which(!where),2*nn,replace=TRUE)
    echantillon0[c((nn+1):nrow(echantillon0)),c("#2 ID","#2 String")] = data_tt[unmatch,c("index","Title")]
    echantillon0[c((nn+1):nrow(echantillon0)),c("Quality")]=0
    }
  
  return(echantillon0)
}
