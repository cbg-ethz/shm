library(tidyverse)
library(gridExtra)
library(grid)
library(ggplot2)


dt <- readr::read_tsv("../shm/data_raw/biological_ternary-data.tsv")
graph <- readr::read_tsv("../shm/data_raw/biological_ternary-graph.tsv")

tt <- ttheme_minimal(base_size=12,
                     base_family="Arial Narrow",
                     core=list(
                       bg_params = list(fill="white")))



dt <- dt %>%
  dplyr::select(gene, condition, intervention, replicate, copynumber, readout)
dt <- dt[1:10, ]
dt[10,] <- "..."

g <- tableGrob(as.data.frame(dt), theme=tt)
plot(g)
ggsave(g,
       filename="~/personal/Sci/phd/presentations/2019_05_29_group_meeting/fig/data_raw.svg",
       width=7.5, height=3, dpi=1000,)

s <- as.tibble(s) %>%
  tidyr::gather(sample, value)
