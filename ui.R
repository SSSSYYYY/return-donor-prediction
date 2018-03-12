library(shiny)
library(shinythemes)

shinyUI(navbarPage(title=strong("BLOOD DONATION"),
                   theme = shinytheme("united"),
        # Tab for introduction
        tabPanel("Introduction",
                 tags$img(src='1.jpg', height = 280,width = 360, align = "right"),
                 h1(strong("Prediction of Blood Donation")),
                 hr(),br(),
                 p(strong("Overview")),
                 p("This app is designed to help blood banks to predict if a person will donate blood."),
                 br()
                 # p(strong("Background")),
                 # p("Blood banks are ....")
                 ),
  
      #Tab for company risk preference
      tabPanel("Risk Preference",
               h1(strong("Before prediction, select your risk preference")),
               hr(),
               # input - company risk preference
               titlePanel("Input - How much you want to get for each condition"),
               br(),
               tags$img(src='2.png', height = 250,width = 350),
               br(),br(),
               p(strong("True Positive(TP):"), "Correctly identify people who will donate blood."),br(),
               p(strong("False Negative(FN):"),"Missed people who will donate blood." ),br(),
               p(strong("False Positive(FP):"),"Incorrectly reported people who won't donate blood."),br(),
               p(strong("True Negative(TN):"),"Correctly identify people who won't donate blood."),
               br(),br(),hr(),
               fluidRow(
                 column(1,br(),br(),br(),
                        p("TP"),br(),br(),p("FN"),br(),br(), p("FP"),br(),br(),p("TN")),
                 column(1,
                        p("Opos"),
                        numericInput("OposTP","",1),
                        numericInput("OposFN","",-1),
                        numericInput("OposFP","",-1),
                        numericInput("OposTN","",1)
                        ),
                 column(1,
                        p("Oneg"),
                        numericInput("OnegTP","",1),
                        numericInput("OnegFN","",-1),
                        numericInput("OnegFP","",-1),
                        numericInput("OnegTN","",1)
                 ),
                 column(1,
                        p("Apos"),
                        numericInput("AposTP","",1),
                        numericInput("AposFN","",-1),
                        numericInput("AposFP","",-1),
                        numericInput("AposTN","",1)
                 ),
                 column(1,
                        p("Aneg"),
                        numericInput("AnegTP","",1),
                        numericInput("AnegFN","",-1),
                        numericInput("AnegFP","",-1),
                        numericInput("AnegTN","",1)
                 ),
                 column(1,
                        p("Bpos"),
                        numericInput("BposTP","",1),
                        numericInput("BposFN","",-1),
                        numericInput("BposFP","",-1),
                        numericInput("BposTN","",1)
                 ),
                 column(1,
                        p("Bneg"),
                        numericInput("BnegTP","",1),
                        numericInput("BnegFN","",-1),
                        numericInput("BnegFP","",-1),
                        numericInput("BnegTN","",1)
                 ),
                 column(1,
                        p("ABpos"),
                        numericInput("ABposTP","",1),
                        numericInput("ABposFN","",-1),
                        numericInput("ABposFP","",-1),
                        numericInput("ABposTN","",1)
                 ),
                 column(1,
                        p("ABneg"),
                        numericInput("ABnegTP","",1),
                        numericInput("ABnegFN","",-1),
                        numericInput("ABnegFP","",-1),
                        numericInput("ABnegTN","",1)
                 )
               ),br(),
               submitButton("Submit", icon("refresh"), width = 100 ),
               
               # Output - Overall cost/benefit matrix
               br(),
               hr(),
               titlePanel("Output - Overall Cost/Benefit Matrix"), 
               br(),
               mainPanel(
                 tableOutput("CnBMatrix"), 
                 br(), br(), br(), br(), br())
               
               ),
        
        
      # Tab for Results
      tabPanel("Results",
             h1(strong("Results of models")),
             hr(),
             
             # Input for results
             fluidRow(
                column(3,
                 p(strong("Input - If perform clustering"))   ,   
                 checkboxInput("cluster","models with clustering(K=3) "),
                 br(),
                 checkboxGroupInput("models","Comparision of models:",
                                    c("ANN","C5.0","CART","Logistic Regression","Logit(10-fold CV)",
                                      "Logit(Bagged)","Logit(Boosted)","LogitBoost",
                                      "LDA","RandomForest","SVM","SVM(5-fold CV)"),
                                    selected=c("LDA","Logistic Regression")),br(),
                 submitButton("Submit", icon("refresh"), width = 100 )
                      ),
                
                column(5,
                      plotOutput("benefitgraph")
                       ),
                
                column(3,
                       tableOutput("stats")
                       )
               )
             
             
             # # Output of stats table
             #  br(),
             #  hr(),
             #  titlePanel("Comparision of Accurancy, Sensitivity, Specificity and AUC"), 
             #  br(),
             #  mainPanel(
             #    dataTableOutput("stats"),
             #  br(), br(), br()   
             #   )
  ),
  
    # Tab for Contact information
    tabPanel("Contact",
           h1(strong("Contact Information")),
           hr(),br(),
           p(strong("Designed by:")),
           p("\t Bubble Tea Rocks"),
           br(),
           p(strong("Team members:")),
           p("Hongxia Shi | Rong (Alice) Liao | Jou Tzu (Rose) Kao | Shenyang Yang | Deepti Bahel"),
           br(),
           p(strong("Contact Information:")),
           p("baheldeepti@gmail.com | shi395@purdue.edu")
  )
))
