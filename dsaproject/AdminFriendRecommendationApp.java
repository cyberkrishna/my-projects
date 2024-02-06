import javafx.application.Application;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;
import org.jgrapht.Graph;
import org.jgrapht.alg.interfaces.VertexScoringAlgorithm;
import org.jgrapht.alg.scoring.PageRank;
import org.jgrapht.graph.DefaultEdge;
import org.jgrapht.graph.SimpleGraph;

import java.util.*;
import java.util.stream.Collectors;

public class AdminFriendRecommendationApp extends Application {
    private UltimateFriendRecommendationSystem recommendationSystem;

    private TextField userTextField;
    private TextField interestTextField;
    private TextArea recommendationTextArea;
    private GraphDisplay graphDisplay;

    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage primaryStage) {
        primaryStage.setTitle("Admin Friend Recommendation App");
        recommendationSystem = new UltimateFriendRecommendationSystem();

        BorderPane borderPane = new BorderPane();

        // Create UI components
        userTextField = new TextField();
        interestTextField = new TextField();
        recommendationTextArea = new TextArea();
        recommendationTextArea.setEditable(false);
        graphDisplay = new GraphDisplay();

        // Buttons
        Button recommendButton = new Button("Get Recommendations");
        recommendButton.setOnAction(e -> onRecommendButtonClick());

        Button addFriendshipButton = new Button("Add Friendship");
        addFriendshipButton.setOnAction(e -> onAddFriendshipButtonClick());

        Button setUserInterestsButton = new Button("Set User Interests");
        setUserInterestsButton.setOnAction(e -> onSetUserInterestsButtonClick());

        Button joinGroupButton = new Button("Join Group");
        joinGroupButton.setOnAction(e -> onJoinGroupButtonClick());

        // Admin Interface
        VBox adminVBox = new VBox(10);
        adminVBox.setPadding(new Insets(10));
        adminVBox.getChildren().addAll(new Label("Admin Interface"), addFriendshipButton, setUserInterestsButton, joinGroupButton);

        // Recommendation Interface
        VBox recommendationVBox = new VBox(10);
        recommendationVBox.setPadding(new Insets(10));
        recommendationVBox.getChildren().addAll(
                new Label("Recommendation Interface"),
                new Label("Enter User:"), userTextField,
                new Label("Enter Interest to Filter (Optional):"), interestTextField,
                recommendButton,
                recommendationTextArea
        );

        // Main layout
        HBox mainHBox = new HBox(20);
        mainHBox.getChildren().addAll(adminVBox, recommendationVBox, graphDisplay);

        // Set up the scene
        Scene scene = new Scene(new BorderPane(mainHBox), 1000, 600);
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    private void onRecommendButtonClick() {
        String user = userTextField.getText().trim();
        String filterInterest = interestTextField.getText().trim();

        if (user.isEmpty()) {
            showAlert("Error", "Please enter a user.");
            return;
        }

        List<Pair<String, Double>> recommendations = recommendationSystem.recommendFriends(user, 5, filterInterest);

        StringBuilder result = new StringBuilder();
        for (Pair<String, Double> recommendation : recommendations) {
            String friend = recommendation.getFirst();
            double score = recommendation.getSecond();
            Set<String> commonInterests = getCommonInterests(user, friend);
            Set<String> commonGroups = getCommonGroups(user, friend);

            result.append(friend).append(" (Score: ").append(score).append(")");

            if (!commonInterests.isEmpty()) {
                result.append(" [Common Interests: ").append(String.join(", ", commonInterests)).append("]");
            }

            if (!commonGroups.isEmpty()) {
                result.append(" [Common Groups: ").append(String.join(", ", commonGroups)).append("]");
            }

            result.append("\n");
        }

        recommendationTextArea.setText(result.toString());

        // Update graph display
        updateGraphDisplay();
    }

    private void onAddFriendshipButtonClick() {
        String user1 = promptUserInput("Enter User 1:");
        String user2 = promptUserInput("Enter User 2:");
        double strength = Double.parseDouble(promptUserInput("Enter Friendship Strength:"));

        if (user1 != null && user2 != null) {
            recommendationSystem.addFriendship(user1.trim(), user2.trim(), strength);
            showAlert("Success", "Friendship added successfully!");
        }
    }

    private void onSetUserInterestsButtonClick() {
        String user = promptUserInput("Enter User:");
        String interestsString = promptUserInput("Enter Interests (comma-separated):");

        if (user != null && interestsString != null) {
            List<String> interests = Arrays.asList(interestsString.trim().split(","));
            recommendationSystem.setUserInterests(user.trim(), interests);
            showAlert("Success", "User interests set successfully!");
        }
    }

    private void onJoinGroupButtonClick() {
        String user = promptUserInput("Enter User:");
        String group = promptUserInput("Enter Group:");

        if (user != null && group != null) {
            recommendationSystem.joinGroup(user.trim(), group.trim());
            showAlert("Success", "User joined the group successfully!");
        }
    }

    private Set<String> getCommonInterests(String user1, String user2) {
        Set<String> interestsUser1 = recommendationSystem.getUserInterests().getOrDefault(user1, Collections.emptySet());
        Set<String> interestsUser2 = recommendationSystem.getUserInterests().getOrDefault(user2, Collections.emptySet());

        Set<String> commonInterests = new HashSet<>(interestsUser1);
        commonInterests.retainAll(interestsUser2);

        return commonInterests;
    }

    private Set<String> getCommonGroups(String user1, String user2) {
        Set<String> groupsUser1 = recommendationSystem.getGroupMemberships().entrySet().stream()
                .filter(entry -> entry.getValue().contains(user1))
                .map(Map.Entry::getKey)
                .collect(Collectors.toSet());

        Set<String> groupsUser2 = recommendationSystem.getGroupMemberships().entrySet().stream()
                .filter(entry -> entry.getValue().contains(user2))
                .map(Map.Entry::getKey)
                .collect(Collectors.toSet());

        Set<String> commonGroups = new HashSet<>(groupsUser1);
        commonGroups.retainAll(groupsUser2);

        return commonGroups;
    }

    private void updateGraphDisplay() {
        graphDisplay.updateGraph(recommendationSystem.getGraph());
    }

    private String promptUserInput(String prompt) {
        return JOptionPane.showInputDialog(null, prompt);
    }

    private void showAlert(String title, String message) {
        JOptionPane.showMessageDialog(null, message, title, JOptionPane.INFORMATION_MESSAGE);
    }

    public static void main(String[] args) {
        launch(args);
    }

    private static class GraphDisplay extends javafx.scene.layout.Region {
        private final javafx.scene.layout.Pane pane;

        public GraphDisplay() {
            pane = new javafx.scene.layout.Pane();
            getChildren().add(pane);
        }

        public void updateGraph(Graph<String, DefaultEdge> graph) {
            // You can use a library like JGraphX or create a custom graph visualization here
            // This is a simplified example using a simple text representation
            StringBuilder graphText = new StringBuilder();
            graphText.append("Graph:\n");

            for (String vertex : graph.vertexSet()) {
                graphText.append(vertex).append(" -> ").append(graph.edgesOf(vertex)).append("\n");
            }

            System.out.println(graphText.toString());
        }

        @Override
        protected void layoutChildren() {
            pane.autosize();
        }
    }
}
