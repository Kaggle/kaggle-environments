package kore;

/**
 * Don't want to import a full blown json library
 * Our json is simple enough to use this
 */
public class KoreJson {

    public static int getIntFromJson(String raw, String key) {
        return (int)Integer.parseInt(getNumberPartFromJson(raw, key));
    }

    private static String getNumberPartFromJson(String raw, String key) {
        int keyIdx = raw.indexOf(key);
        if (keyIdx < 0) {
            throw new IllegalStateException("couldn't find key in raw");
        }
        String rest = raw.substring(keyIdx + key.length() + 3);
        int end = rest.indexOf(",") > 0 ? rest.indexOf(",") : rest.indexOf("}");
        return rest.substring(0, end);
    }

    public static int getPlayerIdxFromJson(String raw) {
        String key = "'player': ";
        String key2 = "\"player\": ";
        int keyIdx = Math.max(raw.indexOf(key), raw.indexOf(key2));
        if (keyIdx < 0) {
            throw new IllegalStateException("couldn't find key in raw");
        }
        String rest = raw.substring(keyIdx + key.length());
        int end = rest.indexOf(",") > 0 ? rest.indexOf(",") : rest.indexOf("}");
        return (int)Integer.parseInt(rest.substring(0, end));
    }

    public static String getStrFromJson(String raw, String key) {
        int keyIdx = raw.indexOf(key);
        if (keyIdx < 0) {
            throw new IllegalStateException("couldn't find key in raw");
        }
        String rest = raw.substring(keyIdx + key.length() + 4);
        int end = rest.indexOf(",") > 0 ? rest.indexOf(",") : rest.indexOf("}");
        return rest.substring(0, end - 1);
    }

    public static double getDoubleFromJson(String raw, String key) {
        String val = getNumberPartFromJson(raw, key);
        return Double.parseDouble(val);
    }

    private static String getStrArrStrFromJson(String raw, String key) {
        int keyIdx = raw.indexOf(key);
        if (keyIdx < 0) {
            throw new IllegalStateException("couldn't find key in raw");
        }
        String rest = raw.substring(keyIdx + key.length() + 4);
        int end = rest.indexOf("],") > 0 ? rest.indexOf("],") : rest.indexOf("]}");
        return rest.substring(0, end);
    }

    public static String[] getPlayerPartsFromJson(String raw) {
        String key = "players";
        int keyIdx = raw.indexOf(key);
        if (keyIdx < 0) {
            throw new IllegalStateException("couldn't find key in raw");
        }
        String rest = raw.substring(keyIdx + key.length() + 5);
        int end = rest.indexOf("]],") > 0 ? rest.indexOf("]],") : rest.indexOf("]]}");
        return rest.substring(0, end).split("], \\[");
    }

    public static String[] getStrArrFromJson(String raw, String key) {
        String arrStr = getStrArrStrFromJson(raw, key);
        return arrStr.split(", ");
    }

    public static int[] getIntArrFromJson(String raw, String key) {
        String[] arrStrParts = getStrArrStrFromJson(raw, key).split(", ");
        int[] intArr = new int[arrStrParts.length];
        for (int i = 0; i < arrStrParts[i].length(); i++) {
            intArr[i] = Integer.parseInt(arrStrParts[i]);
        }
        return intArr;
    }

    public static double[] getDoubleArrFromJson(String raw, String key) {
        String[] arrStrParts = getStrArrStrFromJson(raw, key).split(", ");
        double[] doubleArr = new double[arrStrParts.length];
        for (int i = 0; i < arrStrParts[i].length(); i++) {
            doubleArr[i] = Double.parseDouble(arrStrParts[i]);
        }
        return doubleArr;
    }
    
    public static boolean containsKey(String raw, String key) {
        return raw.indexOf(key) > -1;
    }
}
