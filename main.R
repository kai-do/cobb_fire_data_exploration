source("packages.R")
source("sql_functions.R")

rPackages <- c("tools", "keyring", "odbc", "DBI", "tidyverse", "torch", "purrr", "tm", "caret", "readr", "tidytext", "data.table", "randomForest", "rfviz")
require_packages(rPackages)

### Import Data ###

sql <- "./sql/narratives.sql"
server <- "DCP-FDGAPP1PRD"
params <- NULL

if (!exists("query_df")) {
  query_df <- query_sql(server, sql, params)
}
### set data size ###

write.csv(query_df,"narrative_data.csv", row.names = TRUE)

size <- 10000

df <- sample_n(query_df, size)


### Inspect Data ###

head(df)

df %>% count(incident_type_code)


# distribution of number of words in narratives
df$narrative %>% 
  strsplit(" ") %>% 
  sapply(length) %>% 
  summary()

counts <- sapply(strsplit(df$narrative, " "), length)
mean(counts) # average tokens per document

max_length <- round(mean(counts) + 2 * sd(counts))
max_length

### Create text corpus ###

create_clean_corpus <- function(df) {
  corpus <- Corpus(VectorSource(df))
  corpus %>%
    tm_map(PlainTextDocument) %>%
    tm_map(stripWhitespace) %>%
    tm_map(tolower) %>%
    tm_map(removePunctuation) %>%
    tm_map(removeWords, stopwords("english")) %>%
    tm_map(stemDocument)
}

narrative_corpus <- create_clean_corpus(df$narrative)

narrative_corpus[[1]][1]

### create term document matrix for text analysis ###

tdm <- TermDocumentMatrix(narrative_corpus)

freq <- findFreqTerms(tdm)

freq

terms_grouped <- tdm[freq,] %>%
  as.matrix() %>%
  rowSums() %>%
  data.frame(Term=freq, Frequency = .) %>%
  arrange(desc(Frequency)) %>%
  mutate(prop_term_to_total_terms=Frequency/nrow(.))

terms_grouped

### create document term matrix ###

dtm <- DocumentTermMatrix(narrative_corpus)

dtm

sparse <- removeSparseTerms(dtm, 0.98)

sparse

### convert target category to matrix ###

incident_types <- df %>% 
  select(incident_type_code) %>%
  mutate(across(.fns = as.integer)) %>%
  mutate(incident_type_code = replace_na(incident_type_code, 0)) %>%
  as.matrix()

unique(incident_types)

### to data frame ###

sparse_narrative_df <- as.data.frame(as.matrix(sparse))
colnames(sparse_narrative_df) <- make.names(colnames(sparse_narrative_df))
sparse_narrative_df$incident_type <- incident_types

sparse_narrative_df


### split train and test ###

set.seed(100)

training_id <- sample.int(nrow(sparse_narrative_df), size = nrow(sparse_narrative_df) * 0.8)
training <- sparse_narrative_df[training_id,]
testing <- sparse_narrative_df[-training_id,]

training$incident_type <- as.factor(training$incident_type)
testing$incident_type <- as.factor(testing$incident_type)

### random forest ###

random_forest_model <- randomForest(incident_type ~ ., data=training)

predict_random_forest = predict(random_forest_model, newdata=testing)

predict_random_forest

View(table(testing$incident_type, predict_random_forest))

### plotting random forest ###

rf_prep <- rf_prep(as.data.frame(sparse_narrative_df$incident_type), as.data.frame(sparse_narrative_df[-sparse_narrative_df$incident_type]))


### Create Torch dataset ###

narrative_dataset <- dataset(
  name = "narrative_dataset",
  
  initialize = function(indices) {
    data <- self$prepare_narrative_data(df[indices, ])
    self$xcat <- data[[1]][[1]]
    self$xnum <- data[[1]][[2]]
    self$y <- data[[2]]
  },
  
  .getitem = function(i) {
    xcat <- self$xcat[i, ]
    xnum <- self$xnum[i, ]
    y <- self$y[i, ]
    
    list(x = list(xcat, xnum), y = y)
  },
  
  .length = function() {
    dim(self$y)[1]
  },
  
  
  prepare_narrative_data = function(input) {
    input <- input %>%
      mutate(across(.fns = as.factor)) 
    
    target_col <- input$incident_type_code %>% 
      as.integer() %>%
      `-`(1) %>%
      as.matrix()
    
    categorical_cols <- input %>% 
      select(narrative) %>%
      mutate(across(.fns = as.integer)) %>%
      as.matrix()
    
    list(list(torch_tensor(categorical_cols), torch_tensor(numerical_cols)),
         torch_tensor(target_col))
  }
)
