using CSV
using DataFrames
using Random
using StatsBase
using DecisionTree
using Flux
using Flux: Tensor

using CSV, DataFrames

# Load ball-by-ball data
ball_by_ball_data = CSV.read("C:\\Users\\Ishan Dasgupta\\Downloads\\IPL Ball-by-Ball 2008-2020.csv\\IPL Ball-by-Ball 2008-2020.csv", DataFrame)

# Load 2022 squad data
squad_2023_data = CSV.read("C:\\Users\\Ishan Dasgupta\\Downloads\\ipl_2023_dataset.csv", DataFrame)
# Extract only the columns we need

# Load 2023 auction data
auction_data = CSV.read("C:\\Users\\Ishan Dasgupta\\Downloads\\ipl_2023_dataset.csv", DataFrame)

# Extract team details of each player
# Create dictionary to map players to teams
players_teams = Dict{String, String}()
for row in eachrow(auction_data)
    players_teams[row["Player Name"]] = row["2023 Squad"]
end

# Map player teams to ball-by-ball data
for row in eachrow(ball_by_ball_data)
    try
        row.batting_team = players_teams[row.batsman]
        row.bowling_team = players_teams[row.bowler]
    catch e
        continue  # skip row if key is not found
    end
end

# Get unique teams
teams = unique(vcat(ball_by_ball_data.batting_team, ball_by_ball_data.bowling_team))

# Create a dictionary of team index mapping
team_index = Dict(zip(teams, 1:length(teams)))

# Convert team names to index values
ball_by_ball_data.batting_team_index = [team_index[t] for t in ball_by_ball_data.batting_team]
ball_by_ball_data.bowling_team_index = [team_index[t] for t in ball_by_ball_data.bowling_team]

# Define input and output features
input_features = [:batting_team_index, :bowling_team_index]
output_feature = :is_wicket

# Split data into training and validation sets
train_data = ball_by_ball_data[1:round(Int, nrow(ball_by_ball_data)*0.8), :]
val_data = ball_by_ball_data[round(Int, nrow(ball_by_ball_data)*0.8)+1:end, :]

# Define the neural network
model = Chain(
  Dense(2, 128, relu),
  Dense(128, 64, relu),
  Dense(64, 1)
)

# Define loss function and optimizer
loss(x, y) = Flux.mse(model(x), y)
optimizer = ADAM()

function train_model(x, y, model, loss, optimizer, epochs)
    data = hcat(x..., y')
    for i in 1:epochs
        Flux.train!(loss, params(model), [(data[:,j],) for j in 1:size(data,2)], optimizer)
    end
end

# Train the model
train_x = [Array(train_data[:, f])' for f in input_features]
train_y = reshape(train_data[:, output_feature], length(train_data[:, output_feature]))
train_model(train_x, train_y, model, loss, optimizer, 1000)



# Define function to predict scores for a given match
function predict_scores(match, model, input_features)
    x = Float32[[match[feature] for feature in input_features]]
    y_pred = model(x)
    return round(Int, y_pred[1])
end
# Compute predicted scores for each match in the validation data
val_data[!,:predicted_runs] = [predict_scores(row, model, input_features) for row in eachrow(val_data)]

# Simulate matches and compute final standings
teams = unique(val_data.batting_team)
standings = DataFrame(Team=teams, Points=zeros(length(teams)), NRR=zeros(length(teams)))
for row in eachrow(val_data)
    batting_team = row.batting_team
    bowling_team = row.bowling_team
    runs_scored = row.predicted_runs
    
    # Simulate match outcome
    if runs_scored > row.total_runs - 1
        winning_team = batting_team
        losing_team = bowling_team
        winning_points = 2
        losing_points = 0
    elseif runs_scored == row.total_runs - 1
        winning_team = batting_team
        losing_team = bowling_team
        winning_points = 1
        losing_points = 1
    else
        winning_team = bowling_team
        losing_team = batting_team
        winning_points = 2
        losing_points = 0
    end
    
    # Update standings
    standings[standings.Team .== winning_team, :Points] += winning_points
    standings[standings.Team .== losing_team, :Points] += losing_points
    standings[standings.Team .== winning_team, :NRR] += (runs_scored / 20) - (row.total_runs / 20)
    standings[standings.Team .== losing_team, :NRR] += (row.total_runs / 20) - (runs_scored / 20)
end

# Sort and display final standings
sort!(standings, [:Points, :NRR], rev=true)
display(standings)
