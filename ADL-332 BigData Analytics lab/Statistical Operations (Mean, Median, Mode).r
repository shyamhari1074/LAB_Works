list1 <- c(1,2,3,3,4,5,6)

cat("Mean of the list is", mean(list1))
cat("\nMedian of the list is", median(list1))

mode <- function(x) {
  freq <- table(x)
  max_freq <- max(freq)

  if (max_freq == 1) {
    return("HAS No mode")
  } else {
    return(names(freq[freq == max_freq]))
  }
}

cat("\nMode of the list is", mode(list1))
