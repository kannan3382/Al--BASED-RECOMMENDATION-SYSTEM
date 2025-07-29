import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;

import java.io.File;
import java.util.List;
import java.util.Scanner;

public class RecommendationEngine {
    public static void main(String[] args) throws Exception {
        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter User ID (1 to 5): ");
        int userId = scanner.nextInt();
        scanner.close();

        DataModel model = new FileDataModel(new File("dataset.csv"));

        UserSimilarity similarity = new PearsonCorrelationSimilarity(model);

        UserNeighborhood neighborhood = new NearestNUserNeighborhood(2, similarity, model);

        Recommender recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);

        List<RecommendedItem> recommendations = recommender.recommend(userId, 3); // Top 3

        System.out.println("\nRecommendations for User " + userId + ":");
        if (recommendations.isEmpty()) {
            System.out.println("No recommendations available.");
        } else {
            for (RecommendedItem item : recommendations) {
                System.out.printf("Item ID: %d | Predicted Score: %.1f%n", item.getItemID(), item.getValue());
            }
        }
    }
}
